import copy
import dataclasses
import random
import warnings
from collections.abc import Mapping
from functools import partial
from typing import Any, Literal, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from torch import Tensor

import tabpackage
from tabpackage import _training, nn, optim
from tabpackage.sklearn import _config, _data
from tabpackage.sklearn._config import (
    DatasetName,
    Ensembles,
    GreedyEnsemble,
    HyperparameterBounds,
    IntUniformDistribution,
    KWArgs,
    LogUniformDistribution,
    MetricName,
    OptionalDistribution,
    OptionalHyperparameterBounds,
    PerModelHyperparameters,
    TopKEnsemble,
    UniformDistribution,
)
from tabpackage.sklearn._data import XLike, YLike, YPredLike


def _cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(
        input.transpose(1, 2), target, reduction='none'
    ).mean(1)


def _mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return (input.squeeze(-1) - target).square().mean(1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PredictorKey:
    ensemble_name: str
    autocast_dtype: torch.dtype | None
    backbone_forward_impl: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RefitEnsemble:
    name: str
    member_weights: Tensor | None
    prediction_transform: tabpackage.PredictionTransform | None


def _make_predictor(
    tabpack: tabpackage.TrainedTabPack,
    predictor_key: _PredictorKey,
    *,
    refit_ensemble: _RefitEnsemble | None = None,
    device: torch.device,
) -> tabpackage.TabPackPredictor:
    if refit_ensemble is None:
        predictor = tabpack.make_predictor(
            ensemble_names=[predictor_key.ensemble_name],
            autocast_dtype=predictor_key.autocast_dtype,
        )
    else:
        assert predictor_key.ensemble_name == refit_ensemble.name

        # Completion order can differ from the recipe and weight order.
        model_ids = tabpack.load_finished_model_ids().sort().values
        predictor = tabpackage.TabPackPredictor(
            model_pack=tabpack._make_model_pack(
                model_ids=model_ids,
                model_steps=tabpack.load_model_steps(model_ids),
            ),
            autocast_dtype=predictor_key.autocast_dtype,
            prediction_transform=tabpack.load_model_prediction_transform(),
            ensembles={
                predictor_key.ensemble_name: tabpackage.TabPackPredictor.EnsembleHead(
                    member_pack_idx=torch.arange(len(model_ids)),
                    member_weights=refit_ensemble.member_weights,
                    prediction_transform=refit_ensemble.prediction_transform,
                )
            },
        )

    predictor.to(device)
    predictor.model_pack.backbone.set_forward_impl(predictor_key.backbone_forward_impl)
    return predictor


# >>> Estimators


class _TabPackEstimator(BaseEstimator):
    _classification: bool

    def __init__(
        self,
        *,
        # Models
        n_models: int | None = None,
        n_blocks: (
            int | IntUniformDistribution | HyperparameterBounds[int] | None
        ) = None,
        d_block: int | IntUniformDistribution | HyperparameterBounds[int] | None = None,
        dropout: (
            float
            | OptionalDistribution[UniformDistribution]
            | OptionalHyperparameterBounds
            | None
        ) = None,
        numerical_embeddings: Literal['none', 'linear-relu', 'cosine'] = 'cosine',
        numerical_embeddings_d: (
            int | IntUniformDistribution | HyperparameterBounds[int] | None
        ) = None,
        numerical_embeddings_init_scale: (
            float | LogUniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        # Training
        optimizer: Literal['adamw', 'muon-adamw'] = 'muon-adamw',
        adamw_lr: (
            float | LogUniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        adamw_beta1: (
            float | UniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        adamw_beta2: (
            float | UniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        adamw_eps: (
            float | LogUniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        muon_lr: (
            float | LogUniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        muon_momentum: (
            float | UniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        muon_newton_schulz_steps: int | None = None,
        muon_nesterov: bool | None = None,
        muon_skip_zero_updates: bool | None = None,
        weight_decay: (
            float | LogUniformDistribution | HyperparameterBounds[float] | None
        ) = None,
        batch_size: int | None = None,
        max_epochs: (
            int | IntUniformDistribution | HyperparameterBounds[int] | None
        ) = None,
        early_stopping_patience: int | None = 16,
        # Per-model hyperparameters
        per_model_hyperparameters: list[PerModelHyperparameters] | None = None,
        # Evaluation
        eval_datasets: list[DatasetName] | None = None,
        eval_metrics: list[MetricName] | None = None,
        eval_criterion_dataset: DatasetName | None = None,
        eval_criterion_metric: MetricName | Literal['none'] | None = None,
        eval_batch_size: int | None = None,
        # Ensembles
        ensembles: (
            Ensembles | dict[str, GreedyEnsemble | TopKEnsemble | KWArgs] | None
        ) = None,
        ensemble_eval_datasets: list[DatasetName] | None = None,
        ensemble_eval_metrics: list[MetricName] | None = None,
        ensemble_eval_criterion_dataset: DatasetName | None = None,
        ensemble_eval_criterion_metric: MetricName | Literal['none'] | None = None,
        stop_when_all_ensembles_are_finalized: bool = True,
        # Reproducibility
        random_state: int | np.random.RandomState | None = None,
        # Efficiency
        max_pack_size: int | None = None,
        device: str | torch.device | Literal['auto'] = 'auto',
        aux_device: str | torch.device | Literal['auto'] = 'auto',
        move_datasets_to_device: bool | list[DatasetName] = False,
        autocast_dtype: Literal['bfloat16'] | None = None,
        compile_model: bool = False,
        compile_model_numerical_embeddings: bool = False,
        compile_model_first_block: bool = False,
        compile_optimizer_muon_newton_schulz: bool = False,
        backbone_forward_impl: Literal['indexed', 'dense'] = 'dense',
        # Printing
        print_progress: bool = True,
        show_progress_bar: bool | None = None,
        # Prediction
        prediction_ensemble: str | None = None,
        prediction_batch_size: int | None = None,
        prediction_autocast_dtype: Literal['bfloat16'] | None = None,
        prediction_backbone_forward_impl: Literal['indexed', 'dense'] = 'dense',
    ) -> None:
        """TabPack.

        Note:
            In the docs and code, the MLPs are usually referred to as "models".

        TabPack trains many MLPs with different hyperparameters in parallel and
        builds ensembles as they train. Hyperparameters can be fixed or sampled
        for each MLP. Use distributions or sampling bounds to define a search
        space, or `per_model_hyperparameters` to specify the models directly.

        The fitted TabPack provides access to training results through `tabpack_`.
        After fitting, consider using `TabPack.refit` to refit a selected ensemble on
        the full training data for potentially better performance.

        Args:
            n_models: the total number of models to train.
            n_blocks: the number of backbone blocks per model.
            d_block: the width of each backbone block.
            dropout: the dropout rate.
            numerical_embeddings: the embedding method for numerical features.
            numerical_embeddings_d: the embedding size per numerical feature.
            numerical_embeddings_init_scale: the initialization scale for cosine
                embeddings.
            optimizer: the optimization method.
            adamw_lr: the AdamW learning rate.
            adamw_beta1: the decay rate for AdamW's first moment estimate.
            adamw_beta2: the decay rate for AdamW's second moment estimate.
            adamw_eps: the term used for numerical stability in AdamW.
            muon_lr: the Muon learning rate.
            muon_momentum: the Muon momentum.
            muon_newton_schulz_steps: the number of Newton-Schulz iterations in Muon.
            muon_nesterov: whether Muon uses Nesterov momentum.
            muon_skip_zero_updates: whether Muon skips zero updates.
            weight_decay: the weight decay strength.
            batch_size: the number of training examples per model per step.
            max_epochs: the maximum number of training epochs per model.
            early_stopping_patience: the number of epochs without improvement
                tolerated before stopping each model.
            per_model_hyperparameters: hyperparameters for each model, overriding
                the shared values.
            eval_datasets: the datasets used to evaluate individual models.
            eval_metrics: the metrics used to evaluate individual models.
            eval_criterion_dataset: the dataset used to select each model's best
                checkpoint and guide early stopping.
            eval_criterion_metric: the metric used to select each model's best
                checkpoint and guide early stopping.
            eval_batch_size: the initial number of examples per evaluation batch.
            ensembles: the named ensembles to build during training.
            ensemble_eval_datasets: the datasets used to evaluate ensembles.
            ensemble_eval_metrics: the metrics used to evaluate ensembles.
            ensemble_eval_criterion_dataset: the dataset used to select the best
                version of each ensemble and guide its early stopping.
            ensemble_eval_criterion_metric: the metric used to select the best
                version of each ensemble and guide its early stopping.
            stop_when_all_ensembles_are_finalized: whether to end training once all
                ensembles are finalized, even if some models are still training.
            random_state: controls randomness in hyperparameter sampling and training.
            max_pack_size: the maximum number of models trained in parallel.
            device: the device used for training and prediction.
            aux_device: the device used to compute evaluation metrics and store
                predictions and checkpoints.
            move_datasets_to_device: whether to keep all datasets on the training
                device, or the names of specific datasets to keep there.
            autocast_dtype: the precision used for mixed-precision training.
            compile_model: whether to compile the whole model.
            compile_model_numerical_embeddings: whether to compile the numerical
                embeddings.
            compile_model_first_block: whether to compile the first backbone block.
            compile_optimizer_muon_newton_schulz: whether to compile Muon's
                Newton-Schulz computation.
            backbone_forward_impl: the method used to evaluate the backbone.
            print_progress: whether to print a training summary after each epoch.
            show_progress_bar: whether to show training progress within each epoch.
            prediction_ensemble: the ensemble used for prediction. Required when
                training multiple ensembles.
            prediction_batch_size: the number of examples per prediction batch.
            prediction_autocast_dtype: the precision used for mixed-precision
                prediction.
            prediction_backbone_forward_impl: the method used to evaluate the
                backbone during prediction.
        """
        self.n_models = n_models
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.numerical_embeddings = numerical_embeddings
        self.numerical_embeddings_d = numerical_embeddings_d
        self.numerical_embeddings_init_scale = numerical_embeddings_init_scale

        self.optimizer = optimizer
        self.adamw_lr = adamw_lr
        self.adamw_beta1 = adamw_beta1
        self.adamw_beta2 = adamw_beta2
        self.adamw_eps = adamw_eps
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_newton_schulz_steps = muon_newton_schulz_steps
        self.muon_nesterov = muon_nesterov
        self.muon_skip_zero_updates = muon_skip_zero_updates
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

        self.per_model_hyperparameters = per_model_hyperparameters

        self.eval_datasets = eval_datasets
        self.eval_metrics = eval_metrics
        self.eval_criterion_dataset = eval_criterion_dataset
        self.eval_criterion_metric = eval_criterion_metric
        self.eval_batch_size = eval_batch_size

        self.ensembles = ensembles
        self.ensemble_eval_datasets = ensemble_eval_datasets
        self.ensemble_eval_metrics = ensemble_eval_metrics
        self.ensemble_eval_criterion_dataset = ensemble_eval_criterion_dataset
        self.ensemble_eval_criterion_metric = ensemble_eval_criterion_metric
        self.stop_when_all_ensembles_are_finalized = (
            stop_when_all_ensembles_are_finalized
        )

        self.random_state = random_state

        self.max_pack_size = max_pack_size
        self.device = device
        self.aux_device = aux_device
        self.move_datasets_to_device = move_datasets_to_device
        self.autocast_dtype = autocast_dtype
        self.compile_model = compile_model
        self.compile_model_numerical_embeddings = compile_model_numerical_embeddings
        self.compile_model_first_block = compile_model_first_block
        self.compile_optimizer_muon_newton_schulz = compile_optimizer_muon_newton_schulz
        self.backbone_forward_impl = backbone_forward_impl

        self.print_progress = print_progress
        self.show_progress_bar = show_progress_bar

        self.prediction_ensemble = prediction_ensemble
        self.prediction_batch_size = prediction_batch_size
        self.prediction_autocast_dtype = prediction_autocast_dtype
        self.prediction_backbone_forward_impl = prediction_backbone_forward_impl

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, '_is_fitted', False)

    def _make_hyperparameter_configs(
        self,
        *,
        n_numerical_features: int,
        d_out: int,
        rng: random.Random,
    ) -> tuple[KWArgs, KWArgs, list[KWArgs]]:
        unused_hyperparameters = set()
        if self.optimizer == 'adamw':
            unused_hyperparameters.update(
                (
                    'muon_lr',
                    'muon_momentum',
                    'muon_newton_schulz_steps',
                    'muon_nesterov',
                    'muon_skip_zero_updates',
                )
            )
        elif (
            self.muon_newton_schulz_steps is not None
            and self.muon_newton_schulz_steps < 0
        ):
            raise ValueError('`muon_newton_schulz_steps` must be non-negative')

        use_numerical_embeddings = (
            n_numerical_features > 0 and self.numerical_embeddings != 'none'
        )
        if not use_numerical_embeddings:
            unused_hyperparameters.add('numerical_embeddings_d')
        if not use_numerical_embeddings or self.numerical_embeddings != 'cosine':
            unused_hyperparameters.add('numerical_embeddings_init_scale')

        for name in unused_hyperparameters:
            if getattr(self, name) is not None:
                raise ValueError(
                    f'`{name}` does not apply to the current model and optimizer'
                    ' configuration; leave it as `None`'
                )

        model_config: KWArgs = {
            'd_out': d_out,
            'forward_impl': self.backbone_forward_impl,
        }
        if use_numerical_embeddings:
            model_config['numerical_embeddings'] = {
                'type': (
                    nn.CosineEmbeddingsPack
                    if self.numerical_embeddings == 'cosine'
                    else nn.LinearReLUEmbeddingsPack
                ),
            }

        optimizer_config: KWArgs = {
            'type': (
                optim.AdamWPack if self.optimizer == 'adamw' else optim.MuonAdamWPack
            )
        }
        for name in (
            'muon_newton_schulz_steps',
            'muon_nesterov',
            'muon_skip_zero_updates',
        ):
            value = getattr(self, name)
            if value is not None:
                optimizer_config[name] = value

        # >>> Explicit per-model configuration

        if self.per_model_hyperparameters is not None:
            hyperparameter_names = PerModelHyperparameters.__annotations__.keys()
            shared_hyperparameters = {
                name: getattr(self, name)
                for name in hyperparameter_names
                if getattr(self, name) is not None
            }
            for name, value in shared_hyperparameters.items():
                if isinstance(value, (_config.Distribution, dict)):
                    raise TypeError(
                        f'`{name}` must be a scalar when `per_model_hyperparameters` is'
                        ' provided'
                    )

            per_model_hyperparameter_names = {
                name for entry in self.per_model_hyperparameters for name in entry
            }

            required_hyperparameters = {
                'n_blocks',
                'dropout',
                'adamw_lr',
                'weight_decay',
            }
            if self.optimizer == 'muon-adamw':
                required_hyperparameters.add('muon_lr')
            if use_numerical_embeddings:
                required_hyperparameters.add('numerical_embeddings_d')
                if self.numerical_embeddings == 'cosine':
                    required_hyperparameters.add('numerical_embeddings_init_scale')
            missing_hyperparameters = (
                required_hyperparameters
                - shared_hyperparameters.keys()
                - per_model_hyperparameter_names
            )
            if missing_hyperparameters:
                raise ValueError(
                    'Missing explicit hyperparameters:'
                    f' {sorted(missing_hyperparameters)}'
                )

            if self.d_block is None and 'd_block' not in per_model_hyperparameter_names:
                shared_hyperparameters['d_block'] = 384

            def translate(values: Mapping[str, Any], config: KWArgs) -> None:
                for name, value in values.items():
                    if name in unused_hyperparameters:
                        raise ValueError(
                            f'`{name}` does not apply to the current model and'
                            ' optimizer configuration; omit it from'
                            ' `per_model_hyperparameters`'
                        )
                    if value is not None:
                        _config.validate_hyperparameter(name, value)

                    if name in ('n_blocks', 'd_block', 'dropout'):
                        config['model'][name] = value
                    elif name in (
                        'numerical_embeddings_d',
                        'numerical_embeddings_init_scale',
                    ):
                        embedding_key = (
                            'd_embedding'
                            if name == 'numerical_embeddings_d'
                            else 'init_scale'
                        )
                        config['model'].setdefault('numerical_embeddings', {})[
                            embedding_key
                        ] = value
                    elif name == 'max_epochs':
                        config[name] = value
                    else:
                        optimizer_key = (
                            name.removeprefix('adamw_')
                            if self.optimizer == 'adamw'
                            else name
                        )
                        config['optimizer'][optimizer_key] = value

            translate(
                shared_hyperparameters,
                {'model': model_config, 'optimizer': optimizer_config},
            )

            explicit_hyperparameters = []
            for entry in self.per_model_hyperparameters:
                config: KWArgs = {'model': {}, 'optimizer': {}}
                translate(entry, config)
                explicit_hyperparameters.append(config)

            return model_config, optimizer_config, explicit_hyperparameters

        # >>> Hyperparameter resolution

        default_hyperparameters = {
            'n_blocks': IntUniformDistribution(1, 3),
            'd_block': 384,
            'dropout': OptionalDistribution(UniformDistribution(0.0, 0.5)),
            'adamw_lr': LogUniformDistribution(1e-4, 5e-3),
            'weight_decay': LogUniformDistribution(1e-3, 1.0),
            'adamw_beta1': None,
            'adamw_beta2': None,
            'adamw_eps': None,
            'max_epochs': None,
        }
        if use_numerical_embeddings:
            default_hyperparameters['numerical_embeddings_d'] = IntUniformDistribution(
                8, 32
            )
            if self.numerical_embeddings == 'cosine':
                default_hyperparameters['numerical_embeddings_init_scale'] = (
                    LogUniformDistribution(0.01, 10.0)
                )
        if self.optimizer == 'muon-adamw':
            default_hyperparameters['muon_lr'] = LogUniformDistribution(1e-3, 1e-1)
            default_hyperparameters['muon_momentum'] = None

        resolved_hyperparameters = {}
        for name, default in default_hyperparameters.items():
            value = getattr(self, name)
            value = default if value is None else value
            if value is None:
                continue

            if isinstance(value, dict):
                bounds = value.copy()
                if name == 'dropout':
                    optional = bounds.pop('optional')
                    value = UniformDistribution(**bounds)
                    value = OptionalDistribution(value) if optional else value
                elif name in (
                    'n_blocks',
                    'd_block',
                    'numerical_embeddings_d',
                    'max_epochs',
                ):
                    value = IntUniformDistribution(**bounds)
                elif name in ('adamw_beta1', 'adamw_beta2', 'muon_momentum'):
                    value = UniformDistribution(**bounds)
                else:
                    value = LogUniformDistribution(**bounds)

            _config.validate_hyperparameter(name, value)
            resolved_hyperparameters[name] = value

        def sample(name: str) -> int | float:
            value = resolved_hyperparameters[name]
            return (
                value.sample(rng) if isinstance(value, _config.Distribution) else value
            )

        # >>> Optimizer

        per_model_optimizer_hyperparameters = {}
        for name in (
            'adamw_lr',
            'weight_decay',
            'muon_lr',
            'adamw_beta1',
            'adamw_beta2',
            'adamw_eps',
            'muon_momentum',
        ):
            if name not in resolved_hyperparameters:
                continue

            optimizer_key = (
                name.removeprefix('adamw_') if self.optimizer == 'adamw' else name
            )
            if name in ('adamw_lr', 'weight_decay', 'muon_lr') or isinstance(
                resolved_hyperparameters[name], _config.Distribution
            ):
                per_model_optimizer_hyperparameters[optimizer_key] = name
            else:
                optimizer_config[optimizer_key] = resolved_hyperparameters[name]

        # >>> Per-model sampling

        per_model_hyperparameters = []
        for _ in range(64 if self.n_models is None else self.n_models):
            model_hyperparameters = {}
            if use_numerical_embeddings:
                model_hyperparameters['numerical_embeddings'] = {
                    'd_embedding': sample('numerical_embeddings_d'),
                }
                if self.numerical_embeddings == 'cosine':
                    model_hyperparameters['numerical_embeddings']['init_scale'] = (
                        sample('numerical_embeddings_init_scale')
                    )

            model_hyperparameters['n_blocks'] = sample('n_blocks')
            if isinstance(resolved_hyperparameters['d_block'], _config.Distribution):
                model_hyperparameters['d_block'] = sample('d_block')
            model_hyperparameters['dropout'] = sample('dropout')

            hyperparameters = {
                'model': model_hyperparameters,
                'optimizer': {
                    key: sample(name)
                    for key, name in per_model_optimizer_hyperparameters.items()
                },
            }
            if isinstance(
                resolved_hyperparameters.get('max_epochs'), _config.Distribution
            ):
                hyperparameters['max_epochs'] = sample('max_epochs')
            per_model_hyperparameters.append(hyperparameters)

        if not isinstance(resolved_hyperparameters['d_block'], _config.Distribution):
            model_config['d_block'] = resolved_hyperparameters['d_block']

        return model_config, optimizer_config, per_model_hyperparameters

    def _make_train_kwargs(
        self,
        *,
        n_numerical_features: int,
        d_out: int,
        rng: random.Random,
        default_dataset: DatasetName | None,
        labels: dict[DatasetName, np.ndarray],
        device: torch.device,
    ) -> KWArgs:
        batch_size = self.batch_size
        if batch_size is None:
            n_train = len(labels['train'])
            batch_size = (
                32
                if n_train < 2048
                else 64
                if n_train < 4096
                else 128
                if n_train < 8192
                else 256
                if n_train < 32768
                else 512
                if n_train < 131072
                else 1024
            )

        eval_batch_size = self.eval_batch_size
        eval_batch_size = (
            (4096 if device.type == 'cuda' else 512)
            if eval_batch_size is None
            else eval_batch_size
        )

        aux_device = (
            (device if device.type == 'mps' else torch.device('cpu'))
            if self.aux_device == 'auto'
            else _config.resolve_device(self.aux_device)
        )

        eval_criterion = _config.resolve_criterion(
            self.eval_criterion_dataset,
            self.eval_criterion_metric,
            default_dataset=default_dataset,
            default_metric='accuracy' if self._classification else 'mse',
            d_out=d_out,
        )
        eval_metrics = _config.resolve_eval_metrics(
            self.eval_datasets,
            self.eval_metrics,
            eval_criterion=eval_criterion,
            d_out=d_out,
            labels=labels,
        )

        ensembles = self.ensembles
        ensembles = (
            Ensembles({'default': GreedyEnsemble()})
            if ensembles is None
            else (Ensembles(ensembles) if isinstance(ensembles, dict) else ensembles)
        )

        ensemble_eval_criterion = None
        ensemble_eval_metrics = {}
        if ensembles.members:
            ensemble_eval_criterion = _config.resolve_criterion(
                self.ensemble_eval_criterion_dataset,
                self.ensemble_eval_criterion_metric,
                default_dataset=(
                    default_dataset
                    if eval_criterion is None
                    else eval_criterion.dataset
                ),
                default_metric=(
                    None if eval_criterion is None else eval_criterion.metric
                ),
                d_out=d_out,
            )
            ensemble_eval_metrics = _config.resolve_eval_metrics(
                self.ensemble_eval_datasets,
                self.ensemble_eval_metrics,
                eval_criterion=ensemble_eval_criterion,
                d_out=d_out,
                labels=labels,
            )

        ensemble_configs = ensembles.make_config(
            d_out=d_out,
            eval_criterion=ensemble_eval_criterion,
            default_algorithm_dataset=default_dataset,
            labels=labels,
            device=device,
        )

        model_config, optimizer_config, per_model_hyperparameters = (
            self._make_hyperparameter_configs(
                n_numerical_features=n_numerical_features,
                d_out=d_out,
                rng=rng,
            )
        )

        max_epochs = self.max_epochs
        max_epochs = (
            None
            if isinstance(max_epochs, (IntUniformDistribution, dict))
            else max_epochs
        )

        train_kwargs = {
            'model': model_config,
            'prediction_transform': (
                partial(torch.softmax, dim=-1)
                if self._classification
                else partial(torch.squeeze, dim=-1)
            ),
            'loss_fn': _cross_entropy if self._classification else _mse_loss,
            'optimizer': optimizer_config,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'per_model_hyperparameters': per_model_hyperparameters,
            'eval_metrics': eval_metrics,
            'eval_criterion': eval_criterion,
            'eval_batch_size': eval_batch_size,
            'ensembles': ensemble_configs,
            'max_pack_size': self.max_pack_size,
            'device': device,
            'aux_device': aux_device,
            'autocast_dtype': self.autocast_dtype,
            'compile_model': self.compile_model,
            'compile_model_numerical_embeddings': (
                self.compile_model_numerical_embeddings
            ),
            'compile_model_first_block': self.compile_model_first_block,
            'compile_optimizer_muon_newton_schulz': (
                self.compile_optimizer_muon_newton_schulz
            ),
            'print_progress': self.print_progress,
            'show_progress_bar': self.show_progress_bar,
        }

        if self.n_models is not None:
            train_kwargs['n_models'] = self.n_models
        if eval_criterion is not None:
            train_kwargs['early_stopping_patience'] = self.early_stopping_patience
        if ensemble_configs:
            train_kwargs.update(
                ensemble_eval_metrics={
                    dataset: {
                        name: partial(_config.compute_scalar_metric, metric_fn)
                        for name, metric_fn in dataset_metrics.items()
                    }
                    for dataset, dataset_metrics in ensemble_eval_metrics.items()
                },
                ensemble_eval_criterion=ensemble_eval_criterion,
                stop_when_all_ensembles_are_finalized=self.stop_when_all_ensembles_are_finalized,
            )

        return train_kwargs

    def _resolve_prediction_ensemble(self, ensemble_names: list[str]) -> str | None:
        if self.prediction_ensemble is None:
            if len(ensemble_names) > 1:
                raise ValueError(
                    'Set prediction_ensemble when using multiple ensembles'
                )
            return ensemble_names[0] if ensemble_names else None

        if self.prediction_ensemble not in ensemble_names:
            raise ValueError('Unknown prediction ensemble')
        return self.prediction_ensemble

    def _get_predictor_key(self, ensemble_name: str) -> _PredictorKey:
        return _PredictorKey(
            ensemble_name=ensemble_name,
            autocast_dtype=_config.resolve_autocast_dtype(
                self.prediction_autocast_dtype
            ),
            backbone_forward_impl=self.prediction_backbone_forward_impl,
        )

    def _set_fitted_data(
        self,
        feature_schema: _data.FeatureSchema,
        label_schema: _data.LabelSchema,
        feature_encoder: OrdinalEncoder | None,
    ) -> None:
        self._feature_schema = feature_schema
        self._label_schema = label_schema
        self._feature_encoder = feature_encoder

        self.n_features_in_ = feature_schema.n_features
        if feature_schema.columns is None:
            if hasattr(self, 'feature_names_in_'):
                del self.feature_names_in_
        else:
            self.feature_names_in_ = np.array(feature_schema.columns, dtype=object)

        if label_schema.classes is not None:
            self.classes_ = label_schema.classes

    def fit(
        self,
        X: XLike,
        y: YLike,
        *,
        extra_set: tuple[XLike, YLike] | dict[DatasetName, tuple[XLike, YLike]],
    ) -> Self:
        """Train models and build ensembles using the extra data for validation.

        Args:
            X: the training features.
            y: the training labels or regression targets.
            extra_set: extra data as an `(X, y)` pair or a collection of named
                pairs. Use these datasets to evaluate models and build ensembles.
                When passing named datasets, choose the validation dataset through
                `eval_criterion_dataset`.

        Returns:
            The fitted estimator.
        """
        # >>> Data validation

        feature_schema = _data.FeatureSchema.from_data(X)
        label_schema = _data.LabelSchema.from_data(
            y, classification=self._classification
        )
        if len(X) != len(y):
            raise ValueError('Features and labels must have the same number of rows')

        default_dataset = None if isinstance(extra_set, dict) else 'extra'
        extra_datasets = (
            extra_set if isinstance(extra_set, dict) else {'extra': extra_set}
        )
        if 'train' in extra_datasets:
            raise ValueError('The dataset name "train" is reserved')

        for X_extra, y_extra in extra_datasets.values():
            if _data.FeatureSchema.from_data(X_extra) != feature_schema:
                raise ValueError(
                    'Extra datasets must have the same feature schema as the training'
                    ' data'
                )
            label_schema.validate(y_extra)
            if len(X_extra) != len(y_extra):
                raise ValueError(
                    'Features and labels must have the same number of rows'
                )

        # >>> Encoding

        feature_encoder = feature_schema.make_encoder(X)
        arrays = {
            name: {
                **feature_schema.encode(features, feature_encoder),
                'y': label_schema.encode(target),
            }
            for name, (features, target) in {'train': (X, y), **extra_datasets}.items()
        }

        # >>> Training configuration

        device = _config.resolve_device(self.device)
        base_seed = int(
            check_random_state(self.random_state).randint(0, 2**32 - 2, dtype=np.int64)
        )
        rng = random.Random(base_seed)
        train_kwargs = self._make_train_kwargs(
            n_numerical_features=feature_schema.n_numerical_features,
            d_out=1 if label_schema.classes is None else len(label_schema.classes),
            rng=random.Random(_training.get_next_seed(rng)),
            default_dataset=default_dataset,
            labels={name: data['y'] for name, data in arrays.items()},
            device=device,
        )
        train_kwargs['model']['categorical_cardinalities'] = (
            []
            if feature_encoder is None
            else [len(categories) for categories in feature_encoder.categories_]
        )
        prediction_ensemble = self._resolve_prediction_ensemble(
            list(train_kwargs['ensembles'])
        )
        predictor_key = (
            None
            if prediction_ensemble is None
            else self._get_predictor_key(prediction_ensemble)
        )

        # >>> Dataset placement

        move_datasets_to_device = self.move_datasets_to_device
        datasets_on_device = (
            set(arrays)
            if move_datasets_to_device is True
            else (
                set()
                if move_datasets_to_device is False
                else set(move_datasets_to_device)
            )
        )
        if not datasets_on_device <= arrays.keys():
            raise ValueError('Unknown dataset in move_datasets_to_device')

        datasets = {
            name: _data.make_dataset(
                data,
                device=(
                    device if name in datasets_on_device else train_kwargs['aux_device']
                ),
            )
            for name, data in arrays.items()
        }
        del arrays

        # >>> Training

        fit_train_kwargs = copy.deepcopy(train_kwargs)
        train_device = _config.resolve_device(datasets['train']['y'].device)
        tabpack = tabpackage.train(
            **train_kwargs,
            datasets=datasets,
            seed=_training.get_next_seed(rng),
        )
        predictor = (
            None
            if predictor_key is None
            else _make_predictor(tabpack, predictor_key, device=device)
        )

        # >>> Fitted state

        # Preserve prior fitted state if training or predictor construction fails.
        self._set_fitted_data(feature_schema, label_schema, feature_encoder)
        self.tabpack_ = tabpack
        self._fitted_device = device
        self._predictor = predictor
        self._predictor_key = predictor_key
        self._default_prediction_batch_size = train_kwargs['eval_batch_size']
        self._fit_train_kwargs = fit_train_kwargs
        self._refit_seed_rng = rng
        self._fit_train_device = train_device
        self._refit_ensemble: _RefitEnsemble | None = None
        self._is_fitted = True

        return self

    def refit(
        self,
        X: XLike,
        y: YLike,
        *,
        ensemble: str | None = None,
        max_n_models: int | None = None,
        n_repeats: int | None = None,
    ) -> Self:
        """Retrain an ensemble from scratch on full training data."""
        check_is_fitted(self)
        if self._refit_ensemble is not None:
            raise RuntimeError('Refit requires a fresh fit')

        feature_schema = _data.FeatureSchema.from_data(X)
        label_schema = _data.LabelSchema.from_data(
            y, classification=self._classification
        )
        if len(X) != len(y):
            raise ValueError('Features and labels must have the same number of rows')

        if self._classification:
            assert label_schema.classes is not None
            assert self._label_schema.classes is not None
            if not np.array_equal(label_schema.classes, self._label_schema.classes):
                raise ValueError('Refit requires the same classification labels')

        ensemble_names = self.tabpack_.get_ensemble_names()
        if ensemble is None:
            if len(ensemble_names) != 1:
                raise ValueError(
                    'Omitting ensemble requires exactly one fitted ensemble'
                )
            ensemble = ensemble_names[0]
        else:
            if ensemble not in ensemble_names:
                raise ValueError('Unknown refit ensemble')

        if max_n_models is not None and n_repeats is not None:
            raise ValueError('Specify either max_n_models or n_repeats, not both')

        # >>> Checkpoint recipes

        member_ids = self.tabpack_.load_ensemble_member_ids(ensemble)
        member_weights = self.tabpack_.load_ensemble_member_weights(ensemble)
        n_repeats = (
            n_repeats if max_n_models is None else max_n_models // len(member_ids)
        )
        n_repeats = 1 if n_repeats is None else n_repeats
        if n_repeats <= 0:
            raise ValueError('Refit must include at least one complete ensemble')

        n_steps_per_epoch = self.tabpack_.get_n_steps_per_epoch()
        member_steps = self.tabpack_.load_ensemble_member_steps(ensemble).tolist()
        assert all(step % n_steps_per_epoch == 0 for step in member_steps)
        member_epochs = [step // n_steps_per_epoch for step in member_steps]

        per_model_hyperparameters = self.tabpack_.load_per_model_hyperparameters(
            member_ids.tolist()
        )
        assert per_model_hyperparameters is not None
        for hyperparameters, epochs in zip(
            per_model_hyperparameters, member_epochs, strict=True
        ):
            hyperparameters['max_epochs'] = epochs
            if feature_schema.n_numerical_features == 0:
                hyperparameters.get('model', {}).pop('numerical_embeddings', None)

        refit_ensemble = _RefitEnsemble(
            name=ensemble,
            member_weights=(
                None
                if member_weights is None
                else member_weights.repeat(n_repeats) / n_repeats
            ),
            prediction_transform=self.tabpack_.load_ensemble_prediction_transform(
                ensemble
            ),
        )

        # >>> Training configuration and datasets

        feature_encoder = feature_schema.make_encoder(X)

        train_kwargs = copy.deepcopy(self._fit_train_kwargs)
        train_kwargs['model']['categorical_cardinalities'] = (
            []
            if feature_encoder is None
            else [len(categories) for categories in feature_encoder.categories_]
        )

        if feature_schema.n_numerical_features == 0:
            train_kwargs['model'].pop('numerical_embeddings', None)

        for name in (
            'n_models',
            'early_stopping_patience',
            'ensemble_eval_metrics',
            'ensemble_eval_criterion',
            'stop_when_all_ensembles_are_finalized',
        ):
            train_kwargs.pop(name, None)
        train_kwargs.update(
            max_epochs=None,
            per_model_hyperparameters=[
                copy.deepcopy(hyperparameters)
                for _ in range(n_repeats)
                for hyperparameters in per_model_hyperparameters
            ],
            eval_metrics={},
            eval_criterion=None,
            ensembles={},
        )
        predictor_key = self._get_predictor_key(ensemble)
        datasets = {
            'train': _data.make_dataset(
                {
                    **feature_schema.encode(X, feature_encoder),
                    'y': label_schema.encode(y),
                },
                device=self._fit_train_device,
            )
        }

        # >>> Training

        rng = copy.deepcopy(self._refit_seed_rng)
        with warnings.catch_warnings():
            # Refit can contain fewer models than the original training run.
            warnings.filterwarnings(
                'ignore',
                message=(
                    r'^The provided `max_pack_size=\d+` is no less than'
                    r' the number of models \d+ and therefore will have'
                    r' no effect\. Omit `max_pack_size` to avoid this warning$'
                ),
                category=UserWarning,
            )
            tabpack = tabpackage.train(
                **train_kwargs,
                datasets=datasets,
                seed=_training.get_next_seed(rng),
            )
        predictor = _make_predictor(
            tabpack,
            predictor_key,
            refit_ensemble=refit_ensemble,
            device=train_kwargs['device'],
        )

        # >>> Fitted state

        self._set_fitted_data(feature_schema, label_schema, feature_encoder)
        self.tabpack_ = tabpack
        self._fitted_device = train_kwargs['device']
        self._predictor = predictor
        self._predictor_key = predictor_key
        self._default_prediction_batch_size = train_kwargs['eval_batch_size']
        self._refit_ensemble = refit_ensemble
        self.prediction_ensemble = ensemble

        return self

    def _predict(self, X: XLike) -> YPredLike:
        check_is_fitted(self)
        if _data.FeatureSchema.from_data(X) != self._feature_schema:
            raise ValueError(
                'Prediction data must have the same feature schema as the training data'
            )

        prediction_ensemble = self._resolve_prediction_ensemble(
            self.tabpack_.get_ensemble_names()
            if self._refit_ensemble is None
            else [self._refit_ensemble.name]
        )
        if prediction_ensemble is None:
            raise RuntimeError('No ensembles were trained; prediction is unavailable')

        batch_size = self.prediction_batch_size
        batch_size = (
            self._default_prediction_batch_size if batch_size is None else batch_size
        )
        if batch_size <= 0:
            raise ValueError('The prediction batch size must be positive')

        predictor_key = self._get_predictor_key(prediction_ensemble)
        if predictor_key != self._predictor_key:
            predictor = _make_predictor(
                self.tabpack_,
                predictor_key,
                refit_ensemble=self._refit_ensemble,
                device=self._fitted_device,
            )
            self._predictor = predictor
            self._predictor_key = predictor_key

        dataset = _data.make_dataset(
            self._feature_schema.encode(X, self._feature_encoder),
            device=torch.device('cpu'),
        )

        assert self._predictor is not None
        prediction_batches = []
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                batch = {
                    name: values[start : start + batch_size].to(self._fitted_device)
                    for name, values in dataset.items()
                }
                _, predictions = self._predictor(batch.get('x_num'), batch.get('x_cat'))
                prediction_batches.append(
                    predictions[prediction_ensemble].cpu().numpy()
                )

        return np.concatenate(prediction_batches)


class TabPackClassifier(ClassifierMixin, _TabPackEstimator):
    """A TabPack classifier."""

    _classification = True

    def predict_proba(self, X: XLike) -> YPredLike:
        """Predict class probabilities, with columns in the order of `classes_`."""
        return self._predict(X)

    def predict(self, X: XLike) -> YPredLike:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class TabPackRegressor(RegressorMixin, _TabPackEstimator):
    """A TabPack regressor."""

    _classification = False

    def predict(self, X: XLike) -> YPredLike:
        """Predict regression targets."""
        return self._predict(X)
