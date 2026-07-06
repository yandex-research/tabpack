import dataclasses
import datetime
import gc
import inspect
import itertools
import json
import math
import statistics
import time
import typing
from collections.abc import Callable, Generator, Mapping, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, Protocol, TypedDict

import delu
import numpy as np
import optuna
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib.data
import lib.env
import lib.experiment
import lib.optim.utils
import lib.tools.tune
import lib.utils
import project.ensemble_utils
import project.ensemble_utils_torch
import project.metrics
import project.nn
import project.optim
import project.utils
from lib.types import AMPDType, JSONDict, KWArgs, PartKey, PredictionType, TaskType
from project.metrics import Metrics
from project.nn import BATCH_DIM, PACK_DIM, get_pack_size
from project.types import ConfigDict


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Model
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _make_num_module(type: str, **kwargs) -> nn.Module:
    classes = (
        project.nn.LinearEmbeddingsPack,
        project.nn.LinearReLUEmbeddingsPack,
        project.nn.CosineEmbeddingsPack,
    )
    cls = {x.__name__: x for x in classes}[type]
    return cls(**kwargs)


class ModelPack(project.nn.ModulePack):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        pack_size: int,
        num_embeddings: None | KWArgs = None,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        if num_embeddings is None:
            self.num_module = None
            d_num_feature = 1
        else:
            self.num_module = _make_num_module(
                n_features=n_num_features, pack_size=pack_size, **num_embeddings
            )
            d_num_feature = num_embeddings.get('max_d_embedding')
            if d_num_feature is None:
                d_num_feature = num_embeddings['d_embedding']
            assert isinstance(d_num_feature, int)
        self.cat_module = (
            project.nn.OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        self.pack_view = project.nn.PackView(pack_size=pack_size)
        self.backbone = project.nn.MLPBackbonePack(
            d_in=d_num_feature * n_num_features + sum(cat_cardinalities),
            pack_size=pack_size,
            **backbone_kwargs,
        )
        self.output = project.nn.LinearPack(
            backbone_kwargs['d_block'],
            1 if n_classes is None or n_classes == 2 else n_classes,
            max_in_features=backbone_kwargs.get('max_d_block'),
            pack_size=pack_size,
        )

        self._n_num_features = n_num_features
        self._cat_cardinalities = cat_cardinalities

    @property
    def pack_size(self) -> int:
        return self.backbone.pack_size

    def forward(
        self, x_num: None | Tensor, x_cat: None | Tensor, pack_idx: None | Tensor = None
    ) -> Tensor:
        assert x_num is not None or x_cat is not None

        x_list: list[Tensor] = []
        pack_view_used = False

        if x_num is None:
            assert self._n_num_features == 0
            assert self.num_module is None
        else:
            assert self._n_num_features > 0
            if self.num_module is None:
                x_list.append(x_num)
            else:
                x_num_ = self.pack_view(x_num, pack_idx)
                pack_view_used = True
                x_num_ = self.num_module(x_num_, pack_idx)
                x_list.append(x_num_.flatten(-2))
                del x_num_

        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x_cat_ = self.cat_module(x_cat).to(torch.get_default_dtype())
            if pack_view_used:
                x_cat_ = self.pack_view(x_cat_, pack_idx)
            x_list.append(x_cat_)
            del x_cat_

        x: Tensor = torch.cat(x_list, dim=-1)
        if not pack_view_used:
            x = self.pack_view(x, pack_idx)
        x = self.backbone(x, pack_idx)
        x = self.output(x, pack_idx)
        return x


class ApplyModel(Protocol):
    def __call__(
        self,
        model: nn.Module,
        dataset: lib.data.Dataset,
        *,
        part: PartKey,
        batch_idx: Tensor,
    ) -> Tensor: ...


def apply_model_impl(
    model: nn.Module, dataset: lib.data.Dataset, *, part: PartKey, batch_idx: Tensor
) -> Tensor:
    return (
        model(
            dataset.data['x_num'][part][batch_idx] if 'x_num' in dataset.data else None,
            dataset.data['x_cat'][part][batch_idx] if 'x_cat' in dataset.data else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression predictions.
        .float()
    )


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Hyperparameter sampler
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class HyperparameterSampler:
    """A simpler wrapper around `optuna.study.Study`."""

    _WARMUP_SEED_SHIFT = 1000
    _BASIC_SAMPLERS = (
        'BruteForceSampler',
        'GridSampler',
        'RandomSampler',
        'QMCSampler',
    )

    @staticmethod
    def _make_sampler(type: str, **kwargs) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, type)(**kwargs)

    def __init__(
        self,
        *,
        space: dict[str, Any],
        type: str = 'TPESampler',
        strict_n_startup_trials: bool = False,
        study_kwargs: None | KWArgs = None,
        **kwargs,
    ) -> None:
        self._study = optuna.create_study(
            sampler=HyperparameterSampler._make_sampler(type, **kwargs),
            **({} if study_kwargs is None else study_kwargs),
        )
        self._strict_n_startup_trials = strict_n_startup_trials
        self._space = space
        self._trials: dict[int, optuna.trial.Trial] = {}

    @property
    def n_startup_trials(self) -> None | int:
        return getattr(self._study.sampler, '_n_startup_trials', None)

    def _load_n_finished_trials(self) -> int:
        return len(
            self._study.get_trials(
                False,
                (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
            )
        )

    def ask(self, index: int) -> dict[str, Any]:
        assert index not in self._trials
        trial = self._study.ask()
        if self.n_startup_trials is not None:
            trial.set_user_attr(
                'startup', self._load_n_finished_trials() < self.n_startup_trials
            )
        self._trials[index] = trial
        return lib.tools.tune._sample_config(trial, self._space, [])

    def tell(self, index: int, value: float) -> bool:
        trial = self._trials[index]
        should_skip = (
            self._strict_n_startup_trials
            and self.n_startup_trials is not None
            and trial.user_attrs.get('startup', False)
            and self._load_n_finished_trials() >= self.n_startup_trials
        )
        should_tell = not should_skip
        if should_tell:
            self._study.tell(self._trials[index], value)
        return should_tell


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Pack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _find_arrays_and_tensors(x: Any) -> Generator[np.ndarray | Tensor]:
    if isinstance(x, np.ndarray | Tensor):
        yield x
    elif x is None or x is ... or isinstance(x, bool | int | float | str | bytes):
        return
    elif isinstance(x, Sequence):
        for item in x:
            yield from _find_arrays_and_tensors(item)
    elif isinstance(x, Mapping):
        for item in itertools.chain(x.keys(), x.values()):
            yield from _find_arrays_and_tensors(item)
    elif hasattr(x, '__dict__'):
        yield from _find_arrays_and_tensors(vars(x))


class StatePack:
    _BEST_STEP_INIT = np.iinfo(np.int64).min

    def __init__(self, *, pack_size: int, configs: None | list[ConfigDict]) -> None:
        assert pack_size >= 0

        # Static properties.
        self._ids = np.arange(pack_size)
        self._configs = deepcopy(configs)

        # Dynamic properties.
        self._steps = np.zeros(pack_size, dtype=np.int64)
        self._n_consequtive_bad_updates = np.zeros(pack_size, dtype=np.int64)
        self._best_metrics = {}
        self._best_steps = np.full(pack_size, StatePack._BEST_STEP_INIT)
        self._best_predictions = {}
        self._best_predictions_torch = {}
        self._best_model_state_dicts = {}

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def configs(self) -> None | list[dict]:
        return self._configs

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @property
    def n_consequtive_bad_updates(self) -> np.ndarray:
        return self._n_consequtive_bad_updates

    @property
    def best_metrics(self) -> dict[PartKey, Metrics]:
        return self._best_metrics

    @property
    def best_steps(self) -> np.ndarray:
        return self._best_steps

    @property
    def best_predictions(self) -> dict[PartKey, np.ndarray]:
        return self._best_predictions

    @property
    def best_predictions_torch(self) -> dict[PartKey, Tensor]:
        return self._best_predictions_torch

    @property
    def best_model_state_dicts(self) -> dict[str, Tensor]:
        return self._best_model_state_dicts

    @property
    def pack_size(self) -> int:
        return len(self.ids)

    def validate(self) -> None:
        # Validate the pack size.
        pack_size = self.pack_size
        if self.configs is not None:
            assert len(self.configs) == pack_size
        assert all(
            x.shape[PACK_DIM] == pack_size for x in _find_arrays_and_tensors(self)
        )

        # Validate the static properties.
        assert np.all(self.ids >= 0)
        assert len(np.unique(self.ids)) == pack_size

        # Validate the dynamic properties.
        assert np.all(self.steps >= 0)
        assert np.all(self.n_consequtive_bad_updates >= 0)
        assert np.all(self.best_steps <= self.steps)

    # NOTE
    # All the following methods MUTATE `self`.

    def step(self) -> None:
        self._steps += 1

    def remove(self, pack_idx: np.ndarray) -> None:
        assert len(pack_idx) > 0
        assert (self.steps[pack_idx] > 0).all()

        device = next(iter(self.best_model_state_dicts.values())).device
        keep_pack_idx_torch = project.nn.make_keep_pack_idx(
            self.pack_size, torch.as_tensor(pack_idx)
        )
        keep_pack_idx = keep_pack_idx_torch.numpy()
        keep_pack_idx_torch = keep_pack_idx_torch.to(device)
        del pack_idx

        # NOTE
        # All static and dynamic properties corresponding to `pack_idx`
        # must be removed.

        # Remove the static properties.
        self._ids = self.ids[keep_pack_idx].copy()
        if self.configs is not None:
            self.configs[:] = [self.configs[i] for i in map(int, keep_pack_idx)]

        # Remove the dynamic properties.
        self._steps = self.steps[keep_pack_idx].copy()
        self._n_consequtive_bad_updates = self.n_consequtive_bad_updates[
            keep_pack_idx
        ].copy()
        for part_metrics in self.best_metrics.values():
            for key in list(part_metrics):
                part_metrics[key] = part_metrics[key][keep_pack_idx].copy()
        self._best_steps = self.best_steps[keep_pack_idx].copy()
        for part in list(self.best_predictions):
            self.best_predictions[part] = self.best_predictions[part][
                keep_pack_idx
            ].copy()
        for part in list(self.best_predictions):
            self.best_predictions_torch[part] = self.best_predictions_torch[part][
                keep_pack_idx_torch
            ].clone()
        for key, value in list(self.best_model_state_dicts.items()):
            self.best_model_state_dicts[key] = value[keep_pack_idx_torch]

    def update(
        self,
        metrics: dict[PartKey, Metrics],
        *,
        predictions: dict[PartKey, np.ndarray],
        predictions_torch: dict[PartKey, Tensor],
        model_state_dict: dict[str, Tensor],
    ) -> None:
        assert (self.steps > 0).all()

        if self.best_metrics:
            device = next(iter(model_state_dict.values())).device
            improved_mask = metrics['val']['score'] > self.best_metrics['val']['score']
            improved_pack_idx = np.nonzero(improved_mask)[0]

            if len(improved_pack_idx) > 0:
                improved_pack_idx_torch = torch.tensor(improved_pack_idx, device=device)
                self.n_consequtive_bad_updates[improved_pack_idx] = 0

                for part in self.best_metrics:
                    for key in self.best_metrics[part]:
                        self.best_metrics[part][key][improved_pack_idx] = metrics[part][
                            key
                        ][improved_pack_idx]
                self.best_steps[improved_pack_idx] = self.steps[improved_pack_idx]
                for part in self.best_predictions:
                    self.best_predictions[part][improved_pack_idx] = predictions[part][
                        improved_pack_idx
                    ]
                for part in self.best_predictions_torch:
                    self.best_predictions_torch[part][improved_pack_idx_torch] = (
                        predictions_torch[part][improved_pack_idx_torch]
                    )
                for key, value in model_state_dict.items():
                    self.best_model_state_dicts[key][improved_pack_idx_torch] = value[
                        improved_pack_idx_torch
                    ]
            self.n_consequtive_bad_updates[~improved_mask] += 1

        else:
            assert np.all(self.n_consequtive_bad_updates == 0)
            self._best_metrics = metrics
            self._best_steps[:] = self.steps
            self._best_predictions = deepcopy(predictions)
            self._best_predictions_torch = deepcopy(predictions_torch)
            self._best_model_state_dicts = deepcopy(model_state_dict)


class FinalStatePack:
    def __init__(self) -> None:
        self._ids = np.array([], dtype=np.int64)
        self._steps = np.array([], dtype=np.int64)
        self._predictions = {}
        self._predictions_torch = {}

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @property
    def predictions(self) -> dict[str, np.ndarray]:
        return self._predictions

    @property
    def predictions_torch(self) -> dict[str, Tensor]:
        return self._predictions_torch

    def __len__(self) -> int:
        return len(self.ids)

    def extend(
        self,
        *,
        ids: np.ndarray,
        steps: np.ndarray,
        predictions: dict[str, np.ndarray],
        predictions_torch: dict[str, Tensor],
    ) -> None:
        self._ids = np.concat([self._ids, ids])
        self._steps = np.concat([self._steps, steps])
        self._predictions = (
            {
                k: np.concat([self.predictions[k], predictions[k]])
                for k in predictions.keys()
            }
            if self.predictions
            else predictions
        )
        self._predictions_torch = (
            {
                k: torch.cat([self.predictions_torch[k], predictions_torch[k]])
                for k in predictions_torch.keys()
            }
            if self.predictions_torch
            else predictions_torch
        )


def pack_validate(
    model: ModelPack, optimizer: torch.optim.Optimizer, state: StatePack
) -> None:
    pack_size = model.pack_size
    assert state.pack_size == pack_size

    for x in itertools.chain(model.parameters(), model.buffers()):
        if isinstance(x, project.nn.ParameterPack | project.nn.BufferPack):
            assert get_pack_size(x) == pack_size

    assert pack_size == state.pack_size
    for x in itertools.chain.from_iterable(
        group['params'] for group in optimizer.param_groups
    ):
        assert get_pack_size(x) == pack_size

    state.validate()


def pack_remove(
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    state: StatePack,
    *,
    pack_idx: np.ndarray,
) -> None:
    assert len(pack_idx) > 0
    pack_idx_torch = torch.as_tensor(pack_idx, device=next(model.parameters()).device)
    old_to_new = project.nn.module_pack_remove(model, pack_idx_torch)
    project.optim.optimizer_pack_remove(optimizer, pack_idx_torch, old_to_new)
    state.remove(pack_idx)


def compute_stop_pack_idx(
    state: StatePack, *, epoch_size: int, n_epochs: int, patience: int
) -> None | np.ndarray:
    early_stop_mask = (
        state.n_consequtive_bad_updates > patience if patience >= 0 else None
    )
    epoch_stop_mask = state.steps // epoch_size >= n_epochs if n_epochs >= 0 else None
    stop_mask = (
        early_stop_mask
        if epoch_stop_mask is None
        else epoch_stop_mask
        if early_stop_mask is None
        else (early_stop_mask | epoch_stop_mask)
    )
    if stop_mask is None:
        return None
    stop_pack_idx = np.nonzero(stop_mask)[0]
    return None if len(stop_pack_idx) == 0 else stop_pack_idx


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Experiments
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class ExperimentDict(TypedDict):
    config: NotRequired[JSONDict]
    report: lib.experiment.Report


def assemble_experiments(
    *,
    state: StatePack,
    final_metrics: dict[PartKey, Metrics],
    pack_idx: np.ndarray,
    time_elapsed: float,
) -> list[ExperimentDict]:
    experiments = []
    for i, index in enumerate(pack_idx):
        experiment = {}
        if state.configs is not None:
            experiment['config'] = state.configs[index]
        experiment['report'] = {
            'id': int(state.ids[index]),
            'best_step': int(state.best_steps[index]),
            'metrics': {
                part: {k: float(v[i]) for k, v in part_metrics.items()}
                for part, part_metrics in final_metrics.items()
            },
            # NOTE
            # The time computed below is not representative of the time
            # that running this one experiment would take in isolation.
            'time': time_elapsed,
        }
        experiments.append(experiment)  # ty:ignore[invalid-argument-type]
    return experiments


def get_experiment_val_score(experiment: ExperimentDict) -> float:
    return experiment['report']['metrics']['val']['score']


def get_best_experiment(
    current_best_experiment: None | ExperimentDict,
    new_experiments: list[ExperimentDict],
) -> tuple[ExperimentDict, bool]:
    assert new_experiments

    best_experiment = current_best_experiment
    del current_best_experiment

    best_experiment_improved = False
    for experiment in new_experiments:
        if best_experiment is None or get_experiment_val_score(
            experiment
        ) > get_experiment_val_score(best_experiment):
            best_experiment = experiment
            best_experiment_improved = True

    assert best_experiment is not None
    return best_experiment, best_experiment_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Ensembles
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class OnlineEnsemble:
    def __init__(
        self,
        *,
        type: str,
        options: None | KWArgs = None,
        algorithm_score_fn: None | project.ensemble_utils_torch.EnsembleScoreFn = None,
        update_type: Literal['final', 'best', 'latest'],
        update_part: PartKey,
        include_current_ensemble_in_pool: bool = False,
        prediction_type: PredictionType,
        score_fn: project.ensemble_utils_torch.EnsembleScoreFn,
        patience: int,
    ) -> None:
        # Algorithm details
        self._ensemble_fn = project.ensemble_utils_torch.get_ensemble_fn(type)
        self._ensemble_fn_kwargs = {} if options is None else options
        self._ensemble_fn_kwargs['score_fn'] = (
            score_fn if algorithm_score_fn is None else algorithm_score_fn
        )
        if type == 'autotopk' or (
            type == 'greedy'
            and options is not None
            and options.get('init_autotopk', False)
        ):
            self._ensemble_fn_kwargs['prediction_type'] = prediction_type
        self._update_type = update_type
        self._include_current_ensemble_in_pool = include_current_ensemble_in_pool

        # Current ensemble
        self._ids = np.array([], dtype=np.int64)
        self._steps = np.array([], dtype=np.int64)
        self._predictions: dict[PartKey, np.ndarray] = {}
        self._predictions_torch: dict[PartKey, Tensor] = {}
        self._weights: None | np.ndarray = None
        self._score: None | float = None

        # Task properties
        self._prediction_type = prediction_type

        # Update settings
        self._update_part = update_part
        self._score_fn = score_fn
        self._patience = patience
        self._remaining_patience = patience

        #
        self._time_elapsed = 0.0

    @property
    def ids(self):
        return self._ids

    @property
    def steps(self):
        return self._steps

    @property
    def predictions(self):
        return self._predictions

    @property
    def weights(self):
        return self._weights

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @property
    def is_running(self) -> bool:
        return self._remaining_patience >= 0

    def _prepare_pool(
        self,
        running_ids: np.ndarray,
        running_steps: np.ndarray,
        running_latest_predictions: dict[str, np.ndarray],
        running_latest_predictions_torch: dict[str, Tensor],
        running_best_predictions: dict[str, np.ndarray],
        running_best_predictions_torch: dict[str, Tensor],
        finished_ids: np.ndarray,
        finished_steps: np.ndarray,
        finished_predictions: dict[str, np.ndarray],
        finished_predictions_torch: dict[str, Tensor],
    ):
        if self._update_type == 'final':
            pool_ids = finished_ids
            pool_steps = finished_steps
            pool_predictions = finished_predictions
            pool_predictions_torch = finished_predictions_torch

        else:
            if self._update_type == 'best':
                running_predictions = running_best_predictions
                running_predictions_torch = running_best_predictions_torch
            else:
                assert self._update_type == 'latest'
                running_predictions = running_latest_predictions
                running_predictions_torch = running_latest_predictions_torch

            if len(finished_ids) == 0:
                pool_ids = running_ids
                pool_steps = running_steps
                pool_predictions = running_predictions
                pool_predictions_torch = running_predictions_torch

            else:
                pool_ids = np.concat([finished_ids, running_ids])
                pool_steps = np.concat([finished_steps, running_steps])
                pool_predictions = {
                    k: np.concat([finished_predictions[k], running_predictions[k]])
                    for k in running_predictions.keys()
                }
                pool_predictions_torch = {
                    k: torch.cat(
                        [finished_predictions_torch[k], running_predictions_torch[k]]
                    )
                    for k in running_predictions_torch.keys()
                }

            del running_predictions, running_predictions_torch
        del finished_predictions, finished_predictions_torch
        del running_best_predictions, running_best_predictions_torch
        del running_latest_predictions, running_latest_predictions_torch

        if self._include_current_ensemble_in_pool and len(self.ids) > 0:
            pool_ids = np.concat([self._ids, pool_ids])
            pool_steps = np.concat([self._steps, pool_steps])
            if self._predictions:
                pool_predictions = {
                    k: np.concat([self._predictions[k], pool_predictions[k]])
                    for k in pool_predictions.keys()
                }
            if self._predictions_torch:
                pool_predictions_torch = {
                    k: torch.cat(
                        [self._predictions_torch[k], pool_predictions_torch[k]]
                    )
                    for k in pool_predictions_torch.keys()
                }

        return pool_ids, pool_steps, pool_predictions, pool_predictions_torch

    def update(self, **kwargs) -> bool:
        assert self.is_running
        if self._update_type == 'final' and len(kwargs['finished_ids']) == 0:
            return False

        start_time = time.perf_counter()

        (pool_ids, pool_steps, pool_predictions, pool_predictions_torch) = (
            self._prepare_pool(**kwargs)
        )
        ensemble = self._ensemble_fn(
            pool_predictions_torch[self._update_part], **self._ensemble_fn_kwargs
        )
        ensemble_idx, ensemble_weights = (
            (ensemble, None) if isinstance(ensemble, Tensor) else ensemble
        )

        ensemble_part_prediction = (
            project.ensemble_utils_torch.compute_ensemble_prediction(
                pool_predictions_torch[self._update_part][ensemble_idx],
                ensemble_weights,
                self._prediction_type,
            )
        )
        score = self._score_fn(ensemble_part_prediction[None]).item()
        improved = self._score is None or score > self._score

        if improved:
            self._remaining_patience = self._patience

            ensemble_idx_numpy = ensemble_idx.cpu().numpy()
            self._ids = pool_ids[ensemble_idx_numpy]
            self._steps = pool_steps[ensemble_idx_numpy]
            self._predictions = {
                k: v[ensemble_idx_numpy] for k, v in pool_predictions.items()
            }
            self._predictions_torch = {
                k: v[ensemble_idx] for k, v in pool_predictions_torch.items()
            }
            self._weights = (
                ensemble_weights
                if ensemble_weights is None
                else ensemble_weights.cpu().numpy()
            )
            self._score = score

        else:
            self._remaining_patience -= 1

        self._time_elapsed += time.perf_counter() - start_time
        return improved


def update_online_ensembles(
    online_ensembles: dict[str, OnlineEnsemble],
    *,
    task: lib.data.Task,
    step: int,
    timer: delu.tools.Timer,
    **kwargs,
) -> tuple[dict[str, lib.experiment.Report], bool]:
    reports = {}
    first_online_ensemble_improved = False
    for i_ensemble, (ensemble_name, ensemble) in enumerate(online_ensembles.items()):
        if not ensemble.is_running:
            continue

        ensemble_improved = ensemble.update(**kwargs)
        if i_ensemble == 0:
            first_online_ensemble_improved = ensemble_improved

        if ensemble_improved:
            ensemble_predictions = {
                k: project.ensemble_utils.compute_ensemble_prediction(
                    v, ensemble.weights, ensemble._prediction_type
                )
                for k, v in ensemble._predictions.items()
            }
            metrics = task.calculate_metrics(
                ensemble_predictions, ensemble._prediction_type
            )

            reports[ensemble_name] = {
                'ids': ensemble.ids.tolist(),
                **(
                    {}
                    if ensemble.weights is None
                    else {'weights': ensemble.weights.tolist()}
                ),
                'step': step,
                'steps': ensemble.steps.tolist(),
                'time': timer.elapsed(),
                'ensemble_time': ensemble.time_elapsed,
                'metrics': metrics,
            }

    return reports, first_online_ensemble_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Training
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def generate_training_batches(
    *,
    train_size: int,
    batch_size: int,
    batch_generator: torch.Generator,
    pack_size: int,
) -> list[Tensor]:
    """Generate training batches for one epoch."""
    random_values = torch.rand(
        (pack_size, train_size),
        generator=batch_generator,
        device=batch_generator.device,
    )
    batches = random_values.argsort(dim=BATCH_DIM).split(batch_size, dim=BATCH_DIM)
    batches = list(batches)
    return batches


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Evaluation
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class _EvaluateOutput(NamedTuple):
    metrics: dict[PartKey, Metrics]
    predictions: dict[PartKey, np.ndarray]
    predictions_torch: dict[PartKey, Tensor]


@lib.utils.adjust_gpu_memory_usage('batch_size')
def _evaluate(
    apply_model: ApplyModel,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: lib.data.Dataset,
    *,
    parts: list[PartKey],
    regression_label_stats: None | lib.data.RegressionLabelStats,
    prediction_type: str | PredictionType,
    batch_size: int,
    device: torch.device,
) -> _EvaluateOutput:
    model.eval()
    del optimizer

    metrics = {}
    predictions = {}
    predictions_torch = {}

    for part in parts:
        y_pred_torch = torch.cat(
            [
                apply_model(model, dataset, part=part, batch_idx=batch_idx)
                for batch_idx in torch.arange(dataset.size(part), device=device).split(
                    batch_size
                )
            ],
            dim=BATCH_DIM,
        )

        if dataset.task.is_regression:
            assert regression_label_stats is not None
            y_pred_torch *= regression_label_stats.std
            y_pred_torch += regression_label_stats.mean

        elif dataset.task.is_binclass:
            y_pred_torch = torch.special.expit(y_pred_torch)

        else:
            assert dataset.task.is_multiclass
            y_pred_torch = torch.special.softmax(y_pred_torch, dim=-1)

        y_pred = y_pred_torch.cpu().numpy()

        assert np.isfinite(y_pred).all()
        metrics[part] = project.metrics.calculate_metrics_pack(
            y_true=dataset.task.labels[part],
            y_pred=y_pred,
            task_type=dataset.task.type_,
            prediction_type=prediction_type,
            score=dataset.task.score,
        )
        predictions[part] = y_pred
        predictions_torch[part] = y_pred_torch

    return _EvaluateOutput(metrics, predictions, predictions_torch)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Utilities
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _free_mps_memory() -> None:
    if torch.mps.is_available():
        gc.collect()
        torch.mps.empty_cache()


def _make_Y_train(dataset: lib.data.Dataset[Tensor]) -> Tensor:
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_multiclass else torch.float
    )
    return Y_train


def _make_autocast(amp_dtype: AMPDType, device: torch.device) -> torch.autocast:
    dtype = lib.utils.get_amp_dtype(amp_dtype, device)
    # It is unclear how to implement gradient scaling,
    # so FP16 is not supported for now.
    assert dtype is torch.bfloat16, 'For now, only "bfloat16" is supported as amp_dtype'
    return torch.autocast(device.type, dtype)


def _make_online_ensembles(
    online_ensemble_configs: dict[str, KWArgs],
    *,
    task: lib.data.Task,
    prediction_type: PredictionType,
    update_part: PartKey,
    device: torch.device,
) -> dict[str, OnlineEnsemble]:
    score_fn = project.ensemble_utils_torch.make_emsemble_score_fn(
        task,
        prediction_type,
        part=update_part,
        device=device,
    )
    loss_score_fn = (
        project.ensemble_utils_torch.make_emsemble_score_fn(
            dataclasses.replace(task, score=lib.data.Score.CROSS_ENTROPY),
            prediction_type,
            part=update_part,
            device=device,
        )
        if task.is_classification
        else None
    )

    online_ensembles = {}
    for name, ensemble_config in online_ensemble_configs.items():
        algorithm_score_fn = ensemble_config.get('algorithm_score_fn')
        if algorithm_score_fn == 'loss':
            ensemble_config['algorithm_score_fn'] = loss_score_fn
        online_ensembles[name] = OnlineEnsemble(
            update_part=update_part,
            prediction_type=prediction_type,
            score_fn=score_fn,
            **ensemble_config,
        )

    return online_ensembles


def _make_muon_scale(linear: project.nn.LinearPack) -> Tensor:
    pack_size = linear.pack_size
    out_f = (
        linear.out_features.float()
        if linear.out_features is not None
        else torch.full(
            (pack_size,),
            float(linear.max_out_features),
            device=linear.weight.device,
        )
    )
    in_f = (
        linear.in_features.float()
        if linear.in_features is not None
        else torch.full(
            (pack_size,),
            float(linear.max_in_features),
            device=linear.weight.device,
        )
    )
    return (out_f / in_f).clamp_(min=1.0).sqrt_()


def _default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: nn.Parameter
):
    return lib.optim.utils.default_zero_weight_decay_condition(
        module_name, module, parameter_name, parameter
    ) or isinstance(module, project.nn.LinearEmbeddingsPack)


def _make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    optimizer_cls = getattr(torch.optim, type, None)
    if optimizer_cls is None:
        optimizer_cls = {
            x.__name__: x
            for x in [project.optim.AdamWPack, project.optim.MuonAdamWPack]
        }[type]
    if 'pack_size' not in inspect.signature(optimizer_cls.__init__).parameters:
        kwargs.pop('pack_size', None)
    return optimizer_cls(**kwargs)


def _make_loss_fn_pack(task_type: TaskType) -> Callable[[Tensor, Tensor], Tensor]:
    base_loss_fn = {
        TaskType.REGRESSION: nn.functional.mse_loss,
        TaskType.BINCLASS: nn.functional.binary_cross_entropy_with_logits,
        TaskType.MULTICLASS: nn.functional.cross_entropy,
    }[task_type]

    def loss_fn_pack(y_pred: Tensor, y_true: Tensor) -> Tensor:
        pack_size = get_pack_size(y_pred)
        losses = base_loss_fn(
            y_pred.flatten(0, 1), y_true.flatten(0, 1), reduction='none'
        )
        losses = losses.unflatten(0, (pack_size, y_true.shape[BATCH_DIM]))
        losses = losses.flatten(BATCH_DIM)
        return losses.mean(BATCH_DIM)

    return loss_fn_pack


def _get_mean_scores(
    current_mean_scores: None | dict[PartKey, float],
    experiments: list[ExperimentDict],
    state: StatePack,
    eval_parts: list[PartKey],
) -> tuple[dict[PartKey, float], bool]:
    scores = {}
    for experiment in experiments:
        for part in eval_parts:
            scores.setdefault(part, []).append(
                experiment['report']['metrics'][part]['score']
            )
    for part in eval_parts:
        scores.setdefault(part, []).extend(
            state.best_metrics[part]['score'][state.steps > 0].tolist()
        )
    assert scores

    mean_scores = {k: statistics.mean(v) for k, v in scores.items()}
    return mean_scores, (
        current_mean_scores is None or mean_scores['val'] > current_mean_scores['val']
    )


def _get_best_scores(
    current_best_experiment: None | ExperimentDict,
    current_best_scores: None | dict[PartKey, float],
    state: StatePack,
) -> tuple[dict[PartKey, float], bool]:
    if current_best_experiment is not None and (
        current_best_scores is None
        or get_experiment_val_score(current_best_experiment)
        > current_best_scores['val']
    ):
        current_best_scores = {
            part: current_best_experiment['report']['metrics'][part]['score']
            for part in state.best_metrics.keys()
        }
        best_scores_improved = True
    else:
        best_scores_improved = False

    if state.pack_size == 0 or (state.steps == 0).all():
        assert current_best_scores is not None
        return current_best_scores, best_scores_improved

    else:
        val_scores = state.best_metrics['val']['score']
        best_index = np.argmax(val_scores)
        if (
            current_best_scores is None
            or val_scores[best_index] > current_best_scores['val']
        ):
            return (
                {
                    part: float(part_metrics['score'][best_index])
                    for part, part_metrics in state.best_metrics.items()
                },
                True,
            )
        else:
            assert current_best_scores is not None
            return current_best_scores, best_scores_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Main
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class Config(TypedDict):
    seed: int
    data: KWArgs

    # Model
    n_models: int
    model: KWArgs

    # Training
    optimizer: KWArgs
    batch_size: int
    n_epochs: int
    patience: int

    # Evaluation
    eval_parts: NotRequired[list[PartKey]]
    eval_batch_size: NotRequired[int]

    # Ensembles
    online_ensembles: NotRequired[KWArgs]

    # Configs
    configs: NotRequired[list[ConfigDict]]
    sampler: NotRequired[KWArgs]

    # Efficiency
    amp_dtype: NotRequired[AMPDType]
    timeout: NotRequired[int]

    # Report
    track_experiments: NotRequired[bool]
    track_best_experiment: NotRequired[bool]
    track_online_ensemble_history: NotRequired[bool]
    track_memory_usage: NotRequired[bool]

    # Output
    save_final_predictions: NotRequired[bool]
    save_all_predictions: NotRequired[bool]


def _validate_config(config: Config) -> None:
    configs = config.get('configs')
    if configs is not None:
        assert len(configs) == config['n_models']

    sampler_config = config.get('sampler')
    if (
        sampler_config is not None
        and sampler_config.get('type') not in HyperparameterSampler._BASIC_SAMPLERS
    ):
        raise ValueError(
            'Given the provided sampler config, pack_size must be set explicitly,'
            ' because it can have non-trivial impact on the results'
        )


def _prepare_configs(
    config: Config, hyperparameter_sampler: None | HyperparameterSampler
) -> None | list[ConfigDict]:
    configs = config.get('configs')
    if configs is None:
        if hyperparameter_sampler is None:
            return None
        else:
            return [hyperparameter_sampler.ask(x) for x in range(config['n_models'])]
    else:
        assert hyperparameter_sampler is None
        return configs


def _prepare_model_config(
    config: Config, model_configs_T: None | dict[str, list[Any]]
) -> ConfigDict:
    def infer_max_dimension_(
        subconfig: ConfigDict,
        key: str,
        value: None | int | list[int],
        value_space: None | list[Any],
        value_list: None | list[int],
    ) -> None:
        max_key = f'max_{key}'
        if value is None:
            if value_space is None:
                assert value_list is not None
                assert value_list
                subconfig[max_key] = max(value_list)

                assert max_key in subconfig, (
                    f'The model argument `{max_key}` cannot be inferred from the'
                    ' config, so it must be provided explicitly'
                )
            else:
                # value_space: ["_tune_", "int", min_value, max_value[, step]]
                subconfig[max_key] = value_space[3]
        else:
            assert max_key not in subconfig, (
                f'When the model argument `{key}` is provided explicitly,'
                f' the model argument `{max_key}` must be omitted, which is not true'
            )
            assert value_space is None, (
                f'The model argument `{key}` is presented as both a specific value and'
                ' as a part of the sampler space, which is not allowed'
            )
            subconfig[max_key] = max(value) if isinstance(value, list) else None

    model_config = deepcopy(config['model'])

    # Infer the maximum feature embedding dimension.
    model_num_embeddings_config = model_config.get('num_embeddings')
    if model_num_embeddings_config is not None:
        d_embedding_value = model_num_embeddings_config.get('d_embedding')
        d_embedding_value_space = (
            config.get('sampler', {})
            .get('space', {})
            .get('model', {})
            .get('num_embeddings', {})
            .get('d_embedding')
        )
        d_embedding_value_list = (
            None
            if model_configs_T is None
            or 'num_embeddings' not in model_configs_T
            or 'd_embedding' not in model_configs_T['num_embeddings'][0]
            else [x['d_embedding'] for x in model_configs_T['num_embeddings']]
        )
        if (
            d_embedding_value is not None
            or d_embedding_value_space is not None
            or d_embedding_value_list is not None
        ):
            infer_max_dimension_(
                model_num_embeddings_config,
                'd_embedding',
                d_embedding_value,
                d_embedding_value_space,
                d_embedding_value_list,
            )

    # Infer the maximum backbone dimensions.
    for key in ['n_blocks', 'd_block']:
        infer_max_dimension_(
            model_config,
            key,
            model_config.get(key),
            config.get('sampler', {}).get('space', {}).get('model', {}).get(key),
            None if model_configs_T is None else model_configs_T.get(key),
        )

    if model_configs_T is not None:
        # Merge the configs to the main config.
        project.utils.dict_merge_recursively(
            model_config,
            (
                model_configs_T
                | {
                    key: project.utils.transpose_list_of_dicts(model_configs_T[key])
                    for key in ('num_embeddings', 'activation')
                    if key in model_configs_T
                }
            ),
        )

    return model_config


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    _validate_config(config)

    # >>> Start
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])

    # >>> Efficiency
    device = lib.utils.get_device()
    logger.info(f'Device: {device}')

    amp_dtype = config.get('amp_dtype')
    if amp_dtype is None:
        autocast = None
    else:
        autocast = _make_autocast(amp_dtype, device)
        logger.info('Enabled AMP')

    timer = delu.tools.Timer()
    timeout = config.get('timeout')

    if config.get('track_memory_usage', False):
        torch.cuda.reset_peak_memory_stats(device)

    # >>> Data
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.n_bin_features == 0
    regression_label_stats = dataset.try_standardize_labels_()
    dataset = dataset.to_torch(device)
    n_classes = dataset.task.try_compute_n_classes()
    Y_train = _make_Y_train(dataset)
    train_size = dataset.size('train')

    # >>> Hyperparameter sampler
    hyperparameter_sampler_config = config.get('sampler')
    if hyperparameter_sampler_config is None:
        hyperparameter_sampler = None
    else:
        hyperparameter_sampler = HyperparameterSampler(
            seed=config['seed'],
            study_kwargs={'study_name': report['function'], 'direction': 'maximize'},
            **hyperparameter_sampler_config,
        )
        logger.info('Using hyperparameter sampler')

    # >>> State
    state = StatePack(
        pack_size=config['n_models'],
        configs=_prepare_configs(config, hyperparameter_sampler),
    )
    logger.debug('Created state')
    final_state = FinalStatePack()
    logger.debug('Created final_state')
    # Save all configs before training
    all_configs = deepcopy(state.configs) if state.configs is not None else None
    configs_T = (
        None
        if state.configs is None
        else project.utils.transpose_list_of_dicts(deepcopy(state.configs))
    )
    logger.debug('Transposed the configs')

    # >>> Model
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=n_classes,
        pack_size=state.pack_size,
        **_prepare_model_config(
            config,
            (
                None
                if configs_T is None or 'model' not in configs_T
                else project.utils.transpose_list_of_dicts(configs_T.pop('model'))
            ),
        ),
    )
    logger.debug('Created the model')
    model.to(device)
    logger.debug('Moved the model to the device')

    # NOTE
    # Predictions must be stored in aggregation-friendly units
    # for ensembling purposes (raw logits do _not_ meet this requirement).
    prediction_type = (
        PredictionType.LABELS if dataset.task.is_regression else PredictionType.PROBS
    )
    apply_model = apply_model_impl if autocast is None else autocast(apply_model_impl)

    # >>> Ensembles
    online_ensemble_configs = config.get('online_ensembles')
    if online_ensemble_configs is None:
        online_ensembles = None
        online_ensemble_history = None
    else:
        online_ensembles = _make_online_ensembles(
            online_ensemble_configs,
            task=dataset.task,
            prediction_type=prediction_type,
            update_part='val',
            device=device,
        )
        online_ensemble_history = {}
    logger.debug('Created the ensembles')

    # >>> Training
    muon_optimizer_parameter_groups = (
        [
            {
                'params': [block.linear.weight],
                'muon': True,
                'muon_scale': _make_muon_scale(block.linear),
            }
            for block in model.backbone._iter_blocks()
        ]
        if config['optimizer']['type'] == 'MuonAdamWPack'
        else []
    )
    optimizer = _make_optimizer(
        params=lib.optim.utils.make_parameter_groups(
            model,
            _default_zero_weight_decay_condition,
            custom_groups=muon_optimizer_parameter_groups,
        ),
        pack_size=model.pack_size,
        **config['optimizer'],  # type: ignore
        **(
            {}
            if configs_T is None or 'optimizer' not in configs_T
            else project.utils.transpose_list_of_dicts(configs_T.pop('optimizer'))
        ),
    )
    logger.debug('Created the optimizer')

    assert not configs_T, (
        'The following fields of the generated configs were not used:'
        f' {", ".join(configs_T)}'
    )
    del configs_T

    loss_fn = _make_loss_fn_pack(dataset.task.type_)
    epoch_size = math.ceil(train_size / config['batch_size'])

    # >>> Evaluation
    # The following order of `torch.inference_mode` and `partial` preserves
    # typing-related hints in VSCode.
    evaluate = torch.inference_mode()(
        partial(
            _evaluate,
            apply_model,
            model,
            optimizer,
            dataset,
            regression_label_stats=regression_label_stats,
            prediction_type=prediction_type,
            device=device,
        )
    )
    logger.debug('Created the evaluation function')

    eval_parts = config.get('eval_parts', ['val', 'test'])
    assert 'val' in eval_parts
    eval_batch_size = config.get('eval_batch_size', 32768)

    # >>> Global state
    step = 0
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    experiments: list[ExperimentDict] = []
    online_ensemble_predictions: None | dict[str, dict[PartKey, np.ndarray]] = (
        None
        if online_ensembles is None
        else {name: {} for name in online_ensembles.keys()}
    )
    logger.debug('Prepared the global state')

    # >>> Numerical logs
    #
    # Numerical log is a list of Python objects of the same type
    # (usually, dictionaries holding numerical data, e.g. arrays, tensors, etc.).
    #
    # During training, simply append new records and avoid expensive operations,
    # such as CUDA-to-CPU conversions. After the training, transform the
    # logs as needed (e.g. to a more memory-efficient format) and save to disk.
    steps_numlog = []
    epochs_numlog = []
    pack_epochs_numlog = []

    # `mean_scores` and `best_scores` are maintained only for printing
    # and rely on incomplete experiments (i.e. on the data from `training_state`).
    mean_scores = None
    mean_scores_improved = False
    best_scores = None
    best_scores_improved = False
    first_online_ensemble_scores = None
    first_online_ensemble_improved = False

    # >>> Report
    report['n_models'] = 0
    if hyperparameter_sampler is not None:
        report['n_trials'] = 0
    report['prediction_type'] = prediction_type.value
    report['epoch_size'] = epoch_size
    if online_ensembles is not None:
        report['online_ensembles'] = typing.cast(
            dict[str, ExperimentDict],
            {k: {'report': {}} for k in online_ensembles.keys()},
        )
    track_experiments = config.get('track_experiments', hyperparameter_sampler is None)
    track_best_experiment = config.get(
        'track_best_experiment',
        # If individual models have different hyperparameters,
        # save the best experiment by default.
        state.configs is not None,
    )
    logger.debug('Filled the report')

    # >>> Training loop
    print()
    timer.run()
    pack_validate(model, optimizer, state)

    logger.debug('Starting the training loop')
    while (
        report['n_models'] < config['n_models']
        and (timeout is None or timer.elapsed() < timeout)
        and (
            online_ensembles is None
            or any(x.is_running for x in online_ensembles.values())
        )
    ):
        # >>> Validation
        n_remaining_models = config['n_models'] - report['n_models']
        assert model.pack_size <= n_remaining_models

        # >>> Training phase
        epoch_training_start_time = time.perf_counter()
        model.train()

        batches = generate_training_batches(
            train_size=train_size,
            batch_size=config['batch_size'],
            batch_generator=batch_generator,
            pack_size=state.pack_size,
        )
        batch_losses = []
        batch_sizes = []
        for batch_idx in tqdm(
            batches,
            desc=str(lib.utils.try_get_relative_path(exp)),
            leave=False,
            disable=not lib.env.is_local(),
        ):
            losses = loss_fn(
                apply_model(model, dataset, part='train', batch_idx=batch_idx),
                Y_train[batch_idx],
            )
            # The scale of the gradients should not depend on the number of models,
            # so the individual losses are summed, not averaged.
            loss = losses.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            state.step()

            loss_detached = loss.detach()
            batch_losses.append(loss_detached)
            batch_sizes.append(batch_idx.shape[BATCH_DIM])
            steps_numlog.append(
                {
                    'step': step,
                    'time': timer.elapsed(),
                    'batch_size': batch_sizes[-1],
                    'loss': loss_detached,
                }
            )

        epoch_training_duration = time.perf_counter() - epoch_training_start_time

        del batches, batch_idx, losses, loss, loss_detached  # pyright: ignore[reportPossiblyUnboundVariable]
        _free_mps_memory()

        if config.get('track_memory_usage', False):
            report['memory-usage'] = torch.cuda.memory_stats(device)
            break

        # >>> Evaluation phase
        epoch_evaluation_start_time = time.perf_counter()

        # Evaluate the models.
        (
            (eval_metrics, eval_predictions, eval_predictions_torch),
            eval_batch_size,
        ) = evaluate(parts=eval_parts, batch_size=eval_batch_size)
        report['eval_batch_size'] = eval_batch_size
        state.update(
            eval_metrics,
            predictions=eval_predictions,
            predictions_torch=eval_predictions_torch,
            model_state_dict=model.state_dict(),
        )

        pack_epochs_numlog.append(
            deepcopy(
                {
                    'step': state.steps,
                    'time': np.full((state.pack_size,), timer.elapsed()),
                    'size': np.array([state.pack_size]),
                    'id': state.ids,
                    'metrics': eval_metrics,
                    **(
                        {'predictions': eval_predictions}
                        if config.get('save_all_predictions', False)
                        else {}
                    ),
                }
            )
        )

        # Determine which models to stop.
        pack_size_before_stopping = model.pack_size
        stop_pack_idx = compute_stop_pack_idx(
            state,
            epoch_size=epoch_size,
            n_epochs=config['n_epochs'],
            patience=config['patience'],
        )

        if stop_pack_idx is None:
            latest_predictions = eval_predictions
            latest_predictions_torch = eval_predictions_torch

        else:
            stop_pack_idx_torch = torch.as_tensor(stop_pack_idx)
            n_models_to_stop = len(stop_pack_idx)

            keep_pack_idx_torch = project.nn.make_keep_pack_idx(
                state.pack_size, stop_pack_idx_torch
            )
            keep_pack_idx = keep_pack_idx_torch.numpy()

            stop_pack_idx_torch = stop_pack_idx_torch.to(device)
            keep_pack_idx_torch = keep_pack_idx_torch.to(device)

            # Filter predictions for ensembles.
            latest_predictions = {
                k: v[keep_pack_idx] for k, v in eval_predictions.items()
            }
            latest_predictions_torch = {
                k: v[keep_pack_idx_torch] for k, v in eval_predictions_torch.items()
            }

            # Evaluate the best checkpoints of the stopped models.
            project.nn.module_pack_load_state_dict(
                model, state.best_model_state_dicts, pack_idx=stop_pack_idx_torch
            )
            with project.nn.module_pack_select(model, stop_pack_idx_torch):
                (
                    (final_metrics, final_predictions, final_predictions_torch),
                    eval_batch_size,
                ) = evaluate(parts=['train', 'val', 'test'], batch_size=eval_batch_size)
            report['eval_batch_size'] = eval_batch_size

            # Assemble the new results.
            final_state.extend(
                ids=state.ids[stop_pack_idx],
                steps=state.best_steps[stop_pack_idx],
                predictions=final_predictions,
                predictions_torch=final_predictions_torch,
            )
            new_experiments = assemble_experiments(
                state=state,
                final_metrics=final_metrics,
                pack_idx=stop_pack_idx,
                time_elapsed=timer.elapsed(),
            )
            experiments.extend(new_experiments)
            if hyperparameter_sampler is not None:
                for experiment in new_experiments:
                    report['n_trials'] += hyperparameter_sampler.tell(
                        experiment['report']['id'],
                        get_experiment_val_score(experiment),
                    )

            # Remove the stopped models.
            pack_remove(model, optimizer, state, pack_idx=stop_pack_idx)

            # After the removal, all existing pack indices become invalid.
            del stop_pack_idx

            # Update the report.
            report['n_models'] += n_models_to_stop
            report['step'] = step
            if track_experiments:
                report['experiments'] = experiments
            if track_best_experiment:
                report['best'], _ = get_best_experiment(
                    report.get('best'), new_experiments
                )
            report['time'] = timer.elapsed()

            # Make the update visible.
            lib.experiment.dump_report(exp, report)

            del final_predictions, final_predictions_torch
            _free_mps_memory()

        del eval_predictions, eval_predictions_torch
        _free_mps_memory()

        # Update online ensembles.
        if online_ensembles is None:
            first_online_ensemble_improved = False
        else:
            ensemble_reports, first_online_ensemble_improved = update_online_ensembles(
                online_ensembles,
                task=dataset.task,
                step=step,
                timer=timer,
                running_ids=state.ids,
                running_steps=state.steps,
                running_best_predictions=state.best_predictions,
                running_best_predictions_torch=state.best_predictions_torch,
                finished_ids=final_state.ids,
                finished_steps=final_state.steps,
                finished_predictions=final_state.predictions,
                finished_predictions_torch=final_state.predictions_torch,
                running_latest_predictions=latest_predictions,
                running_latest_predictions_torch=latest_predictions_torch,
            )

            assert online_ensemble_history is not None
            for ensemble_name, ensemble_report in ensemble_reports.items():
                report['online_ensembles'][ensemble_name]['report'] = ensemble_report
                online_ensemble_history.setdefault(ensemble_name, []).append(
                    deepcopy(ensemble_report)
                )
                if config.get('track_online_ensemble_history', False):
                    report['online_ensembles'][ensemble_name]['report']['history'] = (
                        online_ensemble_history[ensemble_name]
                    )
                del ensemble_name, ensemble_report

        del latest_predictions, latest_predictions_torch
        _free_mps_memory()

        # Compute statistics.
        training_throughput = math.trunc(epoch_size / epoch_training_duration)
        total_training_throughput = training_throughput * pack_size_before_stopping
        epoch_mean_loss = statistics.fmean(
            torch.stack(batch_losses).tolist(), batch_sizes
        )
        if track_experiments:
            mean_scores, mean_scores_improved = _get_mean_scores(
                mean_scores, experiments, state, eval_parts
            )
            del mean_scores_improved  # Currently unused.
        if track_best_experiment:
            best_scores, best_scores_improved = _get_best_scores(
                report.get('best'), best_scores, state
            )
        if online_ensembles is not None:
            # Here, we rely on the ordered nature of Python dicts.
            first_online_ensemble_experiment = next(
                iter(report['online_ensembles'].values())
            )
            first_online_ensemble_scores = {
                part: part_metrics['score']
                for part, part_metrics in (
                    first_online_ensemble_experiment['report']
                    .get('metrics', {})
                    .items()
                )
            }

        epochs_numlog.append(
            {'step': step, 'time': timer.elapsed(), 'loss': epoch_mean_loss}
        )

        # Print metrics.
        epoch_evaluation_duration = time.perf_counter() - epoch_evaluation_start_time
        mean_scores_message = (
            None
            if mean_scores is None
            else ' '.join(
                f'[{part[0]}] {score:.3f}' for part, score in mean_scores.items()
            )
        )
        best_scores_message = (
            None
            if best_scores is None
            else ' '.join(
                f'[{part[0]}*] {score:.3f}' for part, score in best_scores.items()
            )
        )
        first_online_ensemble_scores_message = (
            None
            if first_online_ensemble_scores is None or not first_online_ensemble_scores
            else ' '.join(
                f'[{part[0]}$] {score:.3f}'
                for part, score in first_online_ensemble_scores.items()
            )
        )
        print(
            f'{"$" if first_online_ensemble_improved else " "}'
            f'{"*" if best_scores_improved else " "}'
            f' [E] {step // epoch_size:<3}'
            f' [T] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [L] {epoch_mean_loss / pack_size_before_stopping:.3f}'
            f' [M] {report["n_models"]:<3}'
            f'{"" if mean_scores_message is None else f" {mean_scores_message}"}'
            f'{"" if best_scores_message is None else f" {best_scores_message}"}'
            f'{"" if first_online_ensemble_scores_message is None else f" {first_online_ensemble_scores_message}"}'  # noqa: E501
            f' [it/s] {training_throughput:<3} | {total_training_throughput:<5}'
            # f' [e/t] {epoch_evaluation_duration / epoch_training_duration:.3f}'
        )
        del epoch_evaluation_duration

        # >>> Validation
        pack_validate(model, optimizer, state)

    pack_validate(model, optimizer, state)
    report['time'] = timer.elapsed()

    # >>> Main artifacts
    lib.experiment.dump_checkpoint(
        exp,
        {
            'report': report,
            'step': step,
            'random_state': delu.random.get_state(),
            'batch_generator': batch_generator.get_state(),
            'hyperparameter_sampler': hyperparameter_sampler,
            'timer': timer,
        },
    )

    # NOTE
    # The order of values in pack-related artifacts follows the order of finishing,
    # NOT the order of IDs.

    # Patch experiments to include all configs (even for unfinished models)
    experiments_with_all_configs = []
    for model_id in range(config['n_models']):
        # Try to find finished experiment with this ID
        finished_experiment = next(
            (exp for exp in experiments if exp['report']['id'] == model_id), None
        )

        if finished_experiment is not None:
            experiments_with_all_configs.append(finished_experiment)
        else:
            experiment = {}
            if all_configs is not None:
                experiment['config'] = all_configs[model_id]
            experiments_with_all_configs.append(experiment)

    exp.joinpath('experiments.json').write_text(
        json.dumps(experiments_with_all_configs, indent=4)
    )
    if config.get('save_final_predictions', True):
        np.savez(exp / 'predictions.npz', **final_state.predictions)  # type: ignore

    if online_ensemble_predictions is not None:
        assert online_ensemble_history is not None
        np.savez(
            exp / 'online_ensemble_predictions.npz',
            **lib.utils.flatten_dict(
                {str(i): x for i, x in enumerate(online_ensemble_predictions)}
            ),
        )
        exp.joinpath('online_ensemble_history.json').write_text(
            json.dumps(online_ensemble_history, indent=4)
        )

    # >>> Save numerical logs
    np.savez(
        exp / 'numlog.npz',
        **lib.utils.flatten_dict(
            {
                'steps': project.utils.numpy_stack(
                    project.utils.to_numpy(steps_numlog)  # type: ignore
                ),
                'epochs': project.utils.numpy_stack(
                    project.utils.to_numpy(epochs_numlog)  # type: ignore
                ),
                'pack': {
                    'epochs': project.utils.numpy_concatenate(pack_epochs_numlog),
                },
            }
        ),
    )

    # >>> Finish
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.utils.init()
    lib.experiment.run_cli(main)
