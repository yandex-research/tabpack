"""TabPack training."""

import contextlib
import copy
import dataclasses
import datetime
import enum
import gc
import itertools
import math
import random
import shutil
import sys
import time
import typing
import warnings
from collections.abc import Callable
from typing import Any, Final, Literal, Protocol, Self, TypeAlias

import torch
import torch._functorch.config
import torch.nn as nn  # noqa: PLR0402
from torch import Tensor

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None

try:
    import tqdm
except ImportError:
    tqdm = None


import tabpackage
import tabpackage.ensemble  # Ensembling algorithms
import tabpackage.metrics  # Metric computation for model packs
import tabpackage.nn  # Neural network module packs
import tabpackage.optim  # Optimizers packs
import tabpackage.utils.torch  # Generic PyTorch utilities

KWArgs: TypeAlias = dict[str, Any]
"""Keyword arguments for a callable."""

DatasetName: TypeAlias = str
"""A dataset name."""


# >>> Model IDs

ModelID: TypeAlias = int
"""A unique model identifier."""

_IDS_DEVICE: Final = torch.device('cpu')


# >>> Hyperparameters
#
# Utilities for managing hyperparameter dictionaries.

HyperparameterDict: TypeAlias = dict[str, Any]
"""A dictionary storing arbitrary hyperparameters."""


def transpose_dicts_recursively(dicts: list[dict[Any, Any]]) -> dict[Any, Any]:
    """Turn a list of nested dictionaries into a nested dictionary of lists.

    >>> dicts = [
    ...     {
    ...         'model': {
    ...             'numerical_embeddings': {'d_embedding': 16},
    ...             'n_blocks': 2,
    ...             'd_block': 256,
    ...         },
    ...     },
    ...     {
    ...         'model': {
    ...             'numerical_embeddings': {'d_embedding': 24},
    ...             'n_blocks': 3,
    ...             'd_block': 384,
    ...         },
    ...     },
    ... ]
    >>> transpose_dicts_recursively(dicts)
    {'model': {'numerical_embeddings': {'d_embedding': [16, 24]}, 'n_blocks': [2, 3], 'd_block': [256, 384]}}
    """  # noqa: E501
    if not dicts:
        raise ValueError('Provide at least one dictionary')
    if not all(x.keys() == dicts[0].keys() for x in dicts):
        raise ValueError('All dictionaries must have the same keys at each level')

    result = {}
    for key in dicts[0]:
        values = [x[key] for x in dicts]
        result[key] = (
            transpose_dicts_recursively(values)
            if isinstance(values[0], dict)
            else values
        )
    return result


def merge_dicts_recursively(this: dict, other: dict) -> None:
    """Merge `other` dictionary into `this`, recursively.

    Overlapping keys are forbidden.

    >>> this = {
    ...     'model': {
    ...         'numerical_embeddings': {'type': tabpackage.nn.CosineEmbeddingsPack},
    ...         'd_block': 256,
    ...     },
    ... }
    >>> other = {
    ...     'model': {
    ...         'numerical_embeddings': {'d_embedding': 16},
    ...         'n_block': 2,
    ...     },
    ... }
    >>> merge_dicts_recursively(this, other)
    >>> this
    {'model': {'numerical_embeddings': {'type': <class 'tabpackage.nn.CosineEmbeddingsPack'>, 'd_embedding': 16}, 'd_block': 256, 'n_block': 2}}
    """  # noqa: E501
    for key, other_value in other.items():
        if key in this:
            if isinstance(other_value, dict):
                this_value = this[key]
                if not isinstance(this_value, dict):
                    raise TypeError(
                        'Cannot merge a nested dictionary because the existing value is'
                        ' not a dictionary'
                    )
                merge_dicts_recursively(this_value, other_value)
            else:
                raise RuntimeError()
        else:
            this[key] = other_value


# >>> Model pack


PredictionTransform: TypeAlias = Callable[[Tensor], Tensor]
"""A transformation of predictions for evaluation and inference.

The transformation must be deterministic and act on each prediction independently
regardless of the number of batch-like dimensions. To save training results or
predictors with `torch.save`, use a picklable callable, such as
`functools.partial(torch.softmax, dim=-1)`, rather than a lambda or closure.
"""


def forward(
    model_pack: tabpackage.nn.TabPack,
    batch: dict[str, Tensor],
    *,
    autocast: torch.autocast | None,
) -> Tensor:
    """Apply TabPack under the requested autocast context and return FP32 outputs."""
    with contextlib.nullcontext() if autocast is None else autocast:
        x = model_pack(batch.get('x_num', None), batch.get('x_cat', None))

    # Promote the output dtype so downstream transformations, losses, metrics, and
    # ensemble computations receive FP32 inputs. All these operations, in turn,
    # implicitly rely on the presence of this cast.
    return x.float()


def eval_forward(
    model_pack: tabpackage.nn.TabPack,
    batch: dict[str, Tensor],
    *,
    autocast: torch.autocast | None,
    prediction_transform: PredictionTransform | None,
) -> Tensor:
    """Apply TabPack to a batch of objects during inference."""
    x = forward(model_pack, batch, autocast=autocast)
    if prediction_transform is not None:
        x = prediction_transform(x)
    return x


# >>> Training
#
# Now, the essential components of a training step: optimizers, loss functions, etc.


def _max_epochs_to_max_steps(max_epochs: int | None, *, n_steps_per_epoch: int) -> int:
    """Convert the number of epochs to the maximum number of training steps."""
    assert n_steps_per_epoch > 0
    max_steps_upper_bound = torch.iinfo(torch.int64).max

    if max_epochs is None:
        return max_steps_upper_bound
    else:
        if max_epochs <= 0:
            raise ValueError('The maximum number of epochs must be positive')

        max_steps = max_epochs * n_steps_per_epoch
        if max_steps > max_steps_upper_bound:
            raise OverflowError(
                'The epoch limit requires more training steps than `torch.int64`'
                ' supports'
            )

        return max_steps


class LossFn(Protocol):
    """A function computing per-model training losses."""

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute one loss per pack member.

        The loss receives raw model outputs, _before_ applying any prediction
        transformations.

        Tensor shapes for model packs:

        - Predictions: `(pack_size, batch_size, *prediction_shape)`
        - Labels:      `(pack_size, batch_size, *label_shape)`
        - Output:      `(pack_size,)`
        """
        ...


def make_optimizer_pack(
    model_pack: tabpackage.nn.TabPack,
    *,
    type: (
        # Currently, arbitrary optimizer packs are not allowed. To remove this
        # limitation, the creation of parameter groups and the model replacement
        # machinery must become configurable.
        type[tabpackage.optim.AdamWPack]
        | type[tabpackage.optim.MuonAdamWPack]
        | type[torch.optim.Optimizer]
    ),
    support_replacement: bool = False,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create an optimizer pack for a model pack."""
    cls = type
    del type

    if not issubclass(cls, torch.optim.Optimizer):
        raise TypeError(
            'The optimizer type must be a subclass of `torch.optim.Optimizer`'
        )

    if support_replacement and not issubclass(cls, tabpackage.optim.OptimizerPack):
        raise ValueError('The specified optimizer does not support member replacement')

    # A list of parameter groups where a parameter may be presented multiple times
    # both in one group and in multiple groups.
    parameter_groups_with_duplicates: list[dict[str, Any]] = []

    if issubclass(cls, tabpackage.optim.MuonAdamWPack):
        # Add parameters optimized with Muon. Using Muon to optimize parameter packs
        # with varying per-model shapes requires special care, so the special helper
        # function is used to build separate per-layer parameter groups correctly.
        for i_block, block in enumerate(model_pack.backbone.blocks):
            parameter_group = cls.make_muon_parameter_group(block.linear)
            if i_block == 0:
                # The first block is used by all models, so none of the models has zero
                # updates in this block. Giving an explicit hint about that avoids the
                # extra device-to-host synchronization for this block.
                parameter_group['muon_skip_zero_updates'] = False
            parameter_groups_with_duplicates.append(parameter_group)
            del i_block, block

    # Add parameters protected from the weight decay regularization.
    parameter_groups_with_duplicates.append(
        {
            'params': [
                p
                for mn, m in model_pack.named_modules()
                for pn, p in m.named_parameters()
                if tabpackage.optim.should_skip_weight_decay(mn, m, pn, p)
            ],
            'weight_decay': 0.0,
        },
    )

    # Add the remaining parameters.
    parameter_groups_with_duplicates.append({'params': list(model_pack.parameters())})

    # Keep each parameter in the group where it first appeared, giving Muon and
    # no-weight-decay groups precedence over the fallback group.
    parameters = set()
    parameter_groups = []
    for group in parameter_groups_with_duplicates:
        deduplicated_params = []
        for p in group['params']:
            if p not in parameters:
                parameters.add(p)
                deduplicated_params.append(p)
        group['params'] = deduplicated_params
        if group['params']:
            parameter_groups.append(group)
    del parameter_groups_with_duplicates

    # Build the optimizer.
    return cls(
        params=parameter_groups,
        **(
            {'allow_inplace_grad_mutation': True}
            if issubclass(cls, tabpackage.optim.MuonAdamWPack)
            else {}
        ),
        **(
            {
                'pack_size': model_pack.pack_size,
                'support_replacement': support_replacement,
            }
            if issubclass(cls, tabpackage.optim.OptimizerPack)
            else {}
        ),
        **kwargs,
    )


def make_shuffled_batch_indices_pack(
    *,
    total_size: int,
    batch_size: int,
    pack_size: int,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
) -> list[Tensor]:
    """Generate a pack of random batch sequences."""
    random_values = torch.rand(
        (pack_size, total_size), generator=generator, device=device
    )
    batches = random_values.argsort(dim=tabpackage.nn.BATCH_DIM).split(
        batch_size, dim=tabpackage.nn.BATCH_DIM
    )
    return list(batches)


# >>> Evaluation
#
# Utilities for per-model evaluation.

MetricName: TypeAlias = str


@enum.unique
class MetricDirection(enum.StrEnum):
    """The optimization direction of a metric."""

    MINIMIZE = enum.auto()
    MAXIMIZE = enum.auto()


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvalCriterionConfig:
    """The dataset and metric used to select the best model and ensemble checkpoints.

    Attributes:
        dataset: the name of the dataset on which to compare checkpoints or ensembles.
        metric: the name of the metric to compare. It must be included for this dataset
            in `train`'s `eval_metrics` or `ensemble_eval_metrics`, respectively.
        metric_direction: whether lower or higher metric values are better.
    """

    dataset: DatasetName
    metric: MetricName
    metric_direction: MetricDirection


class MetricFn(Protocol):
    """A function computing a metric for a model or model pack."""

    def __call__(self, *, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Compute a metric from labels and predictions.

        The callable accepts _transformed_ model predictions and the same labels for all
        models. Metrics used to select checkpoints or ensemble members must produce a
        single value for each prediction.

        Tensor shapes for single models:

        - Labels:      `(batch_size, *label_shape)`
        - Predictions: `(batch_size, *prediction_shape)`
        - Output:      `(*metric_shape,)`

        Tensor shapes for model packs:

        - Labels:      `(batch_size, *label_shape)`
        - Predictions: `(pack_size, batch_size, *prediction_shape)`
        - Output:      `(pack_size, *metric_shape,)`
        """
        ...


@torch.inference_mode()
def _evaluate(
    *,
    model_pack: tabpackage.nn.TabPack,
    prediction_transform: PredictionTransform | None = None,
    datasets: dict[DatasetName, dict[str, Tensor]],
    labels: dict[DatasetName, Tensor],
    metric_fns: dict[DatasetName, dict[MetricName, MetricFn]],
    prediction_dataset_names: tuple[DatasetName, ...],
    batch_size: int,
    device: torch.device,
    aux_device: torch.device,
    autocast: torch.autocast | None,
) -> tuple[
    dict[DatasetName, Tensor],  # Predictions
    dict[DatasetName, dict[MetricName, Tensor]],  # Metrics
]:
    assert datasets.keys() == labels.keys()
    assert metric_fns.keys() <= datasets.keys()
    assert len(set(prediction_dataset_names)) == len(prediction_dataset_names)
    assert set(prediction_dataset_names) <= datasets.keys()

    model_pack.eval()
    predictions = {}
    metrics = {}

    for dataset_name in dict.fromkeys([*metric_fns, *prediction_dataset_names]):
        dataset = datasets[dataset_name]
        y_true = labels[dataset_name]
        dataset_size = len(y_true)

        y_pred = torch.cat(
            [
                eval_forward(
                    model_pack,
                    {k: v[i : i + batch_size].to(device) for k, v in dataset.items()},
                    autocast=autocast,
                    prediction_transform=prediction_transform,
                ).to(aux_device)
                for i in range(0, dataset_size, batch_size)
            ],
            dim=tabpackage.nn.BATCH_DIM,
        )
        if dataset_name in prediction_dataset_names:
            predictions[dataset_name] = y_pred

        if dataset_name in metric_fns:
            metrics[dataset_name] = {}
            for metric_name, metric_fn in metric_fns[dataset_name].items():
                metric_value = metric_fn(y_true=y_true, y_pred=y_pred).clone()
                if metric_value.ndim <= 0:
                    raise ValueError(
                        'Metrics must return a tensor with a dimension for the models'
                    )

                if metric_value.shape[tabpackage.nn.PACK_DIM] != model_pack.pack_size:
                    raise ValueError(
                        'The pack dimension of each metric tensor must have one entry'
                        ' per model'
                    )
                metrics[dataset_name][metric_name] = metric_value
        del y_pred

    return predictions, metrics


def evaluate(
    *, batch_size: int, **kwargs: Any
) -> tuple[
    tuple[dict[DatasetName, Tensor], dict[DatasetName, dict[MetricName, Tensor]]],
    int,
]:
    devices = (kwargs['device'], kwargs['aux_device'])

    while True:
        try:
            return _evaluate(batch_size=batch_size, **kwargs), batch_size
        except RuntimeError as err:
            message = str(err).lower()
            if batch_size <= 1 or not (
                isinstance(err, torch.OutOfMemoryError)
                or 'out of memory' in message
                or (
                    any(x.type == 'mps' for x in devices)
                    and (
                        'invalid buffer size:' in message
                        or (
                            'value cannot be converted to type dest_t without overflow'
                            in message
                        )
                        or (
                            any(x in message for x in ('2**32', 'int_max', '32-bit'))
                            and any(
                                x in message
                                for x in (
                                    'not supported',
                                    'larger than',
                                    "can't be indexed",
                                )
                            )
                        )
                    )
                )
            ):
                raise

        gc.collect()
        for device in devices:
            if device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device.type == 'mps' and torch.mps.is_available():
                torch.mps.empty_cache()

        new_batch_size = batch_size // 2
        warnings.warn(
            f'Evaluation batch size {batch_size} exceeds memory or tensor size limits.'
            f' Retrying with the evaluation batch size {new_batch_size}',
            category=RuntimeWarning,
        )
        batch_size = max(1, new_batch_size)


# >>> State
#
# To efficiently manage a pack of models during the run, below, we define:
#
# - Classes storing common state variables of models as stacked tensors.
# - Functions implementing basic training loop operations, such as recording the
#   best-so-far checkpoints of running models or removing finished models from a pack.


# The notion of "checkpoint key" is an internal primitive serving for implementation
# convenience and is not supposed to be used in public API.
@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelCheckpointKey:
    """An identifier of a model checkpoint."""

    id: int
    step: int

    def __post_init__(self) -> None:
        assert self.id >= 0
        assert self.step >= 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelCheckpointKeys:
    """An ordered collection of checkpoint keys."""

    ids: Tensor  # Shape: (n_checkpoints,)
    steps: Tensor  # Shape: (n_checkpoints,)

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.ids)

    def validate(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError('Checkpoint IDs must be one-dimensional')

        if self.ids.dtype != torch.int64:
            raise TypeError('Checkpoint IDs must have the data type `torch.int64`')

        if self.ids.device != _IDS_DEVICE:
            raise ValueError('Checkpoint IDs must be on the CPU')

        if not torch.all(self.ids >= 0):
            raise ValueError('Checkpoint IDs must be non-negative')

        assert self.steps.shape == self.ids.shape
        assert self.steps.dtype == torch.int64
        assert self.steps.device == self.ids.device
        assert torch.all(self.steps >= 0)

    def tolist(self) -> list[ModelCheckpointKey]:
        return [
            ModelCheckpointKey(id=id, step=step)
            for id, step in zip(self.ids.tolist(), self.steps.tolist(), strict=True)
        ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelCheckpoints:
    """Model checkpoints."""

    keys: ModelCheckpointKeys
    state_dicts: dict[str, Tensor]  # Shape: (n_checkpoints, *value_shape)
    predictions: dict[
        DatasetName, Tensor  # Shape: (n_checkpoints, dataset_size, *prediction_shape)
    ]
    metrics: dict[DatasetName, dict[MetricName, Tensor]]  # Shape: (n_checkpoints,)

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.keys)

    def validate(self) -> None:
        self.keys.validate()

        size = len(self)
        assert (
            torch.unique(torch.stack((self.keys.ids, self.keys.steps)), dim=1).shape[1]
            == size
        )

        for x in itertools.chain(
            self.state_dicts.values(),
            self.predictions.values(),
            itertools.chain.from_iterable(x.values() for x in self.metrics.values()),
        ):
            assert len(x) == size, (
                'Checkpoint data must contain one entry per checkpoint'
            )


# The checkpoint store is a list of checkpoint batches.
CheckpointStore: TypeAlias = list[ModelCheckpoints]


def rebuild_checkpoint_store(
    checkpoint_store: CheckpointStore,
    new_checkpoints: list[ModelCheckpoints],
    *,
    required_keys: set[ModelCheckpointKey],
    state_dict_device: torch.device,
    copy_new_checkpoints: bool,
) -> CheckpointStore:
    """Rebuild a checkpoint store with a required subset of current and new checkpoints."""  # noqa: E501
    new_checkpoint_store = []
    new_checkpoint_keys = set()
    for source in [checkpoint_store, new_checkpoints]:
        for checkpoints in source:
            keep_idx = [
                i
                for i, x in enumerate(checkpoints.keys.tolist())
                if x in required_keys and x not in new_checkpoint_keys
            ]
            if keep_idx:
                if len(keep_idx) < len(checkpoints):
                    checkpoints = tabpackage.utils.torch.index(
                        checkpoints, torch.tensor(keep_idx)
                    )
                elif source is new_checkpoints and copy_new_checkpoints:
                    checkpoints = copy.deepcopy(checkpoints)
                new_checkpoint_store.append(
                    dataclasses.replace(
                        checkpoints,
                        state_dicts=tabpackage.utils.torch.to(
                            checkpoints.state_dicts, state_dict_device
                        ),
                    )
                    if source is new_checkpoints
                    else checkpoints
                )
                new_checkpoint_keys.update(checkpoints.keys.tolist())
    assert new_checkpoint_keys == required_keys
    return new_checkpoint_store


def gather_checkpoint_predictions(
    checkpoint_store: CheckpointStore,
    checkpoint_keys: ModelCheckpointKeys,
    dataset_name: DatasetName,
) -> Tensor:
    """Gather checkpoint predictions from the store in the order of the keys."""
    key_to_prediction = {
        key: prediction
        for batch in checkpoint_store
        for key, prediction in zip(batch.keys.tolist(), batch.predictions[dataset_name])
    }
    # The same key may appear in multiple batches, but it always refers to the same
    # checkpoint, so it does not matter which of the batches provides the prediction.
    return torch.stack([key_to_prediction[x] for x in checkpoint_keys.tolist()])


# The ID values must not repeat within one instance of the dataclass.
@dataclasses.dataclass(kw_only=True)
class TrainingStatePack:
    """Training state of a model pack."""

    ids: Tensor  # Shape: `(pack_size,)`
    max_steps: Tensor  # Shape: `(pack_size,)`
    steps: Tensor  # Shape: `(pack_size,)`
    n_consecutive_bad_updates: Tensor  # Shape: `(pack_size,)`
    best_checkpoints: ModelCheckpoints | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def pack_size(self) -> int:
        return len(self.ids)

    def validate(self) -> None:
        assert self.ids.ndim == 1
        assert self.ids.dtype == torch.int64
        assert self.ids.device == _IDS_DEVICE
        assert torch.all(self.ids >= 0)
        assert len(torch.unique(self.ids)) == self.pack_size

        for x in (self.max_steps, self.steps, self.n_consecutive_bad_updates):
            assert x.shape == self.ids.shape
            assert x.dtype == torch.int64
            assert x.device == self.ids.device
            del x

        assert torch.all(self.max_steps > 0)
        assert torch.all(self.steps >= 0)
        assert torch.all(self.n_consecutive_bad_updates >= 0)

        if self.best_checkpoints is not None:
            assert torch.equal(self.best_checkpoints.keys.ids, self.ids)

    def is_running(self, *, early_stopping_patience: int | None) -> Tensor:
        """Determine which models in a pack are still running."""
        running_mask = self.steps < self.max_steps
        if early_stopping_patience is not None:
            assert early_stopping_patience >= 0, (
                'The early stopping patience must be non-negative'
            )
            running_mask &= self.n_consecutive_bad_updates <= early_stopping_patience
        return running_mask

    def replace(
        self, *, pack_idx: Tensor, ids: Tensor, max_steps: Tensor | None = None
    ) -> None:
        """Reset the state at the given positions for new models with the given IDs."""
        tabpackage.nn.check_pack_idx(pack_idx, self.pack_size)
        assert ids.shape == pack_idx.shape
        if max_steps is not None:
            assert max_steps.shape == pack_idx.shape

        self.ids[pack_idx] = ids
        if max_steps is not None:
            self.max_steps[pack_idx] = max_steps
        self.steps[pack_idx] = 0
        self.n_consecutive_bad_updates[pack_idx] = 0

        if self.best_checkpoints is not None:
            self.best_checkpoints.keys.ids[pack_idx] = ids
            self.best_checkpoints.keys.steps[pack_idx] = 0
            for value in itertools.chain(
                self.best_checkpoints.state_dicts.values(),
                self.best_checkpoints.predictions.values(),
                itertools.chain.from_iterable(
                    x.values() for x in self.best_checkpoints.metrics.values()
                ),
            ):
                value[pack_idx] = math.nan if value.is_floating_point() else 0

        self.validate()


def update_training_state(
    training_state: TrainingStatePack,
    *,
    latest_checkpoints: ModelCheckpoints,
    eval_criterion: 'EvalCriterionConfig | None',
) -> None:
    """Update the training state."""
    assert torch.equal(latest_checkpoints.keys.ids, training_state.ids)

    if eval_criterion is None:
        # Without the criterion, any update is an improvement.
        training_state.best_checkpoints = copy.deepcopy(latest_checkpoints)
        return

    latest_eval_criterion_metrics = latest_checkpoints.metrics[eval_criterion.dataset][
        eval_criterion.metric
    ]
    if latest_eval_criterion_metrics.ndim != 1:
        raise ValueError(
            'The metric used as the evaluation criterion must return one scalar per'
            ' model'
        )

    # For a given model, NaN values of the criterion metric always counts as a
    # non-improvement regardless of any other conditions.
    nonimprovable_mask = torch.isnan(latest_eval_criterion_metrics)

    if training_state.best_checkpoints is None:
        training_state.n_consecutive_bad_updates[
            nonimprovable_mask.to(training_state.n_consecutive_bad_updates.device)
        ] += 1
        training_state.best_checkpoints = copy.deepcopy(latest_checkpoints)

    else:
        best_eval_criterion_metrics = training_state.best_checkpoints.metrics[
            eval_criterion.dataset
        ][eval_criterion.metric]
        assert best_eval_criterion_metrics.shape == latest_eval_criterion_metrics.shape

        first_update_mask = training_state.best_checkpoints.keys.steps == 0
        improved_mask = (
            # Any non-NaN value is considered as an improvement over NaN.
            torch.isnan(best_eval_criterion_metrics)
            | (
                latest_eval_criterion_metrics < best_eval_criterion_metrics
                if eval_criterion.metric_direction == MetricDirection.MINIMIZE
                else latest_eval_criterion_metrics > best_eval_criterion_metrics
            )
        ).to(training_state.best_checkpoints.keys.steps.device)
        improved_mask |= first_update_mask
        improved_mask &= ~nonimprovable_mask.to(improved_mask.device)
        training_state.n_consecutive_bad_updates[improved_mask] = 0
        training_state.n_consecutive_bad_updates[~improved_mask] += 1

        # Store the first evaluated checkpoint even if its metric is NaN.
        updated_mask = improved_mask | first_update_mask
        updated_pack_idx = torch.nonzero(updated_mask)[:, 0]
        if len(updated_pack_idx) > 0:
            # The following function call updates all tensors in a given object at given
            # indices, recursively.
            tabpackage.utils.torch.index_set_(
                # Update the current best evaluation data ...
                training_state.best_checkpoints,
                # ... at the position corresponding to the updated models ...
                (updated_pack_idx,),
                # ... with the corresponding slice of the latest evaluation data.
                tabpackage.utils.torch.index(latest_checkpoints, updated_pack_idx),
            )


def pack_remove(
    model_pack: tabpackage.nn.TabPack,
    optimizer_pack: torch.optim.Optimizer,
    training_state: TrainingStatePack,
    *,
    pack_idx: Tensor,
) -> TrainingStatePack:
    """Remove some models from a model pack."""
    assert len(pack_idx) > 0

    pack_idx_on_device = pack_idx.to(next(model_pack.parameters()).device)

    # Remove the stopped models by removing the corresponding slices from parameter and
    # buffer packs. Technically, this means replacing the parameter and buffer packs
    # with smaller ones. The returned mapping links old parameter tensors to new ones.
    old_to_new_parameters = model_pack.remove(pack_idx=pack_idx_on_device)

    # Now, in the optimizer pack, replace the old optimized parameter tensors with new
    # ones, and remove the parts of the optimizer state tensor corresponding to the
    # removed models.
    tabpackage.optim.optimizer_pack_remove(
        optimizer_pack,
        pack_idx=pack_idx_on_device,
        old_to_new_parameters=old_to_new_parameters,
    )

    del pack_idx_on_device

    # Determine which models to _keep_ in the pack.
    keep_pack_idx = tabpackage.nn.make_complementary_pack_idx(
        pack_idx, pack_size=training_state.pack_size
    )

    return tabpackage.utils.torch.index(training_state, keep_pack_idx)


def pack_replace(
    model_pack: tabpackage.nn.TabPack,
    optimizer_pack: tabpackage.optim.OptimizerPack,
    training_state: TrainingStatePack,
    *,
    pack_idx: Tensor,
    ids: list[ModelID],
    per_model_hyperparameters: list[HyperparameterDict],
    n_steps_per_epoch: int,
) -> None:
    """Replace some models in a model pack with new ones."""
    assert len(pack_idx) == len(ids)
    assert len(pack_idx) == len(per_model_hyperparameters)

    pack_idx_on_device = pack_idx.to(next(model_pack.parameters()).device)

    per_model_hyperparameters_T = transpose_dicts_recursively(per_model_hyperparameters)

    # Reinitialize the corresponding slices of the model parameters according to the
    # new model hyperparameters. Note that the pack was created with the upper bounds
    # of the model shape hyperparameters, so any new model fits the existing tensors.
    model_pack.replace(
        pack_idx=pack_idx_on_device, **per_model_hyperparameters_T.pop('model', {})
    )

    # Reset the corresponding slices of the optimizer state and set the new optimization
    # hyperparameters. The optimizer must be updated _after_ the model pack, because the
    # Muon update scales are recomputed from the new model shapes.
    optimizer_pack.replace(  # ty: ignore[unresolved-attribute]
        pack_idx=pack_idx_on_device,
        **(
            {'module': model_pack}
            if isinstance(optimizer_pack, tabpackage.optim.MuonAdamWPack)
            else {}
        ),
        **per_model_hyperparameters_T.pop('optimizer', {}),
    )

    del pack_idx_on_device

    max_epochs = per_model_hyperparameters_T.pop('max_epochs', None)

    # Ensure that all hyperparameters have been used.
    assert not per_model_hyperparameters_T

    # Reset the training state for the replaced models.
    ids_tensor = torch.tensor(
        ids, dtype=training_state.ids.dtype, device=training_state.ids.device
    )
    training_state.replace(
        pack_idx=pack_idx,
        ids=ids_tensor,
        max_steps=(
            None
            if max_epochs is None
            else torch.tensor(
                [
                    _max_epochs_to_max_steps(x, n_steps_per_epoch=n_steps_per_epoch)
                    for x in max_epochs
                ],
                dtype=training_state.max_steps.dtype,
                device=training_state.max_steps.device,
            )
        ),
    )


# >>> Ensembles
#
# The code for on-the-fly ensembling construction.

EnsembleName: TypeAlias = str


EnsembleAlgorithmFn: TypeAlias = Callable[
    # predictions -> (member_idx, member_weights | None)
    [Tensor], tuple[Tensor, Tensor | None]
]


def check_ensemble_member_idx(member_idx: Tensor) -> None:
    if member_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            'Indices of ensemble members must have the data type `torch.int32` or'
            ' `torch.int64`'
        )

    if member_idx.ndim != 1:
        raise ValueError('Ensemble member indices must be one-dimensional')

    if len(member_idx) == 0:
        raise ValueError('Select at least one ensemble member')


def check_ensemble_member_weights(member_weights: Tensor) -> None:
    if not torch.is_floating_point(member_weights):
        raise TypeError('Ensemble member weights must be floating-point values')

    if member_weights.ndim != 1:
        raise ValueError('Ensemble member weights must be one-dimensional')

    if len(member_weights) == 0:
        raise ValueError('Provide at least one ensemble member weight')


def compute_ensemble_predictions(
    member_predictions: Tensor,
    member_weights: Tensor | None,
    prediction_transform: PredictionTransform | None,
) -> Tensor:
    predictions = tabpackage.ensemble.compute_unsanitized_ensemble_predictions(
        member_predictions, member_weights
    )
    if prediction_transform is not None:
        predictions = prediction_transform(predictions)
    return predictions


class Ensemble:
    """An ensemble that can be updated during training with new model predictions."""

    def __init__(
        self,
        *,
        algorithm_fn: EnsembleAlgorithmFn,
        algorithm_dataset: DatasetName,
        online: bool = True,
        prediction_transform: PredictionTransform | None,
        score_fn: Callable[[Tensor], float] | None,
        score_dataset: DatasetName | None,
        early_stopping_patience: int | None,
    ) -> None:
        """
        Args:
            algorithm_fn: the algorithm for building an ensemble from a given pool
                of predictions, and optionally assigning weights to the selected
                predictions.
            algorithm_dataset: which dataset to use to build the ensemble.
            online: if `True`, use current members, best checkpoints of models that have
                finished training, and latest checkpoints of still-training models for
                rebuilding the ensemble after each epoch. Otherwise, use only best
                checkpoints of finished models, also after each epoch.
            prediction_transform: transform the aggregated _ensemble_ predictions
                before scoring (the passed individual model predictions are not
                affected). The function must not change the prediction shape. Pass
                `None` if no transformation is needed.
            score_fn: the scoring function for _ensemble_ predictions used to accept or
                reject new ensembles. Higher scores are better. If `None`, accept every
                rebuilt ensemble without scoring it.
            score_dataset: which dataset to use to score ensemble updates. Must be
                `None` if and only if `score_fn` is `None`.
            early_stopping_patience: how many unsuccessful updates to tolerate before
                finalizing the ensemble. Must be `None` without the scoring function.
        """
        assert (score_fn is None) == (score_dataset is None)
        if early_stopping_patience is not None:
            assert early_stopping_patience >= 0, (
                'The early stopping patience must be non-negative'
            )
            assert score_fn is not None

        self._algorithm_fn: Final = algorithm_fn
        self._algorithm_dataset: Final = algorithm_dataset
        self._online: Final = online
        self._prediction_transform: Final = prediction_transform

        self._score_fn: Final = score_fn
        self._score_dataset: Final = score_dataset
        self._early_stopping_patience: Final = early_stopping_patience
        self._best_score: float | None = None
        self._n_consecutive_bad_updates = 0

        self._member_checkpoint_keys: ModelCheckpointKeys | None = None
        self._member_weights: Tensor | None = None

    @property
    def algorithm_dataset(self) -> DatasetName:
        return self._algorithm_dataset

    @property
    def prediction_transform(self) -> PredictionTransform | None:
        """The transformation applied to aggregated ensemble predictions."""
        return self._prediction_transform

    @property
    def member_checkpoint_keys(self) -> ModelCheckpointKeys | None:
        """The checkpoint keys of the ensemble members."""
        return self._member_checkpoint_keys

    @property
    def member_weights(self) -> Tensor | None:
        """The weights (importance) of the ensemble members."""
        return self._member_weights

    def is_running(self) -> bool:
        """Check if the ensemble is still running and can be further updated."""
        return (
            self._early_stopping_patience is None
            or self._n_consecutive_bad_updates <= self._early_stopping_patience
        )

    def update(
        self,
        *,
        checkpoint_store: CheckpointStore,
        finished_best_checkpoint_keys: ModelCheckpointKeys,
        running_latest_checkpoint_keys: ModelCheckpointKeys,
    ) -> bool:
        """Try to rebuild the ensemble using new predictions.

        Offline updates with no finished models do not consume patience.

        Args:
            checkpoint_store: the store containing eligible input checkpoints and,
                in online mode, current members.
            finished_best_checkpoint_keys: best checkpoints of all finished models.
            running_latest_checkpoint_keys: latest checkpoints of still-running models,
                considered only in online mode.
        Returns:
            `True` if the rebuilt ensemble is accepted, `False` otherwise.
        """
        assert self.is_running()
        assert finished_best_checkpoint_keys or running_latest_checkpoint_keys
        assert len(
            {
                *finished_best_checkpoint_keys.tolist(),
                *running_latest_checkpoint_keys.tolist(),
            }
        ) == len(finished_best_checkpoint_keys) + len(running_latest_checkpoint_keys)

        if self._online:
            candidate_keys = tabpackage.utils.torch.cat(
                [finished_best_checkpoint_keys, running_latest_checkpoint_keys]
            )
            if self.member_checkpoint_keys is not None:
                current_member_checkpoint_key_set = set(
                    self.member_checkpoint_keys.tolist()
                )
                new_candidate_idx = [
                    i
                    for i, key in enumerate(candidate_keys.tolist())
                    if key not in current_member_checkpoint_key_set
                ]
                # Keeping current members first favors them when the selection
                # algorithm breaks ties by candidate order.
                candidate_keys = tabpackage.utils.torch.cat(
                    [
                        self.member_checkpoint_keys,
                        tabpackage.utils.torch.index(candidate_keys, new_candidate_idx),
                    ]
                )

        elif finished_best_checkpoint_keys:
            candidate_keys = finished_best_checkpoint_keys

        else:
            # No finished models are available yet. In the offline mode, this does not
            # count as a bad update.
            return False

        if (
            self._score_fn is not None
            and self.member_checkpoint_keys is not None
            and (
                set(self.member_checkpoint_keys.tolist())
                == set(candidate_keys.tolist())
            )
        ):
            # The candidate pool contains exactly the current members. Skip rebuilding
            # and count this as a bad update.
            self._n_consecutive_bad_updates += 1
            return False

        # Build a new ensemble.
        member_idx, member_weights = self._algorithm_fn(
            gather_checkpoint_predictions(
                checkpoint_store, candidate_keys, self.algorithm_dataset
            )
        )
        check_ensemble_member_idx(member_idx)
        if member_weights is not None:
            if member_idx.shape != member_weights.shape:
                raise ValueError(
                    'The number of member weights does not match the number of ensemble'
                    ' members'
                )
            check_ensemble_member_weights(member_weights)

        member_checkpoint_keys = tabpackage.utils.torch.index(
            candidate_keys, member_idx
        )
        if len(set(member_checkpoint_keys.tolist())) != len(member_checkpoint_keys):
            raise ValueError(
                'The ensemble algorithm selected the same model checkpoint more than'
                ' once'
            )

        if self._score_fn is None:
            score = None
            improved = True
        else:
            # Evaluate the new ensemble.
            assert self._score_dataset is not None
            member_predictions = gather_checkpoint_predictions(
                checkpoint_store, member_checkpoint_keys, self._score_dataset
            )
            ensemble_predictions = compute_ensemble_predictions(
                member_predictions, member_weights, self._prediction_transform
            )
            score = self._score_fn(ensemble_predictions)
            improved = not math.isnan(score) and (
                self._best_score is None or self._best_score < score
            )
        if improved:
            # Replace the current members with new ones.
            self._member_weights = member_weights
            self._member_checkpoint_keys = member_checkpoint_keys
            self._best_score = score
            self._n_consecutive_bad_updates = 0
        else:
            self._n_consecutive_bad_updates += 1

        return improved


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnsembleConfig:
    """Configuration for an ensemble built during training.

    Attributes:
        algorithm: the ensemble algorithm configuration dictionary. The `"type"` entry
            specifies the callable used to select members and optionally assign their
            weights; the remaining entries are keyword arguments for that callable.
            See `EnsembleAlgorithmType` for details.
        online: whether to consider models that are still training when building the
            ensemble. If `True`, each update can select from the current ensemble
            members, the best checkpoints of finished models, and the latest
            checkpoints of still-training models. If `False`, only the best checkpoints
            of finished models are considered. In both cases, updates are attempted
            after each epoch.
        prediction_transform: the transformation applied after combining ensemble
            members' predictions, for evaluation and inference. It must preserve the
            prediction shape. `None` leaves the combined predictions unchanged.
        early_stopping_patience: the number of consecutive updates without improvement
            in `train`'s `ensemble_eval_criterion` tolerated before finalizing the
            ensemble. A finalized ensemble keeps its selected members and is no longer
            updated. `None` disables ensemble-level early stopping. Must be `None` if
            `ensemble_eval_criterion` is not set.
        algorithm_metric: the metric callable used by the algorithm to select ensemble
            members and, if supported, their weights. It must return one scalar per
            candidate prediction in the input pack. See `MetricFn` for details. This
            metric guides ensemble construction; `train`'s `ensemble_eval_criterion`
            determines whether to accept the resulting ensemble.
        algorithm_metric_direction: whether the algorithm should minimize or maximize
            `algorithm_metric`.
        algorithm_dataset: the name of the dataset whose labels and predictions are
            used to build the ensemble. Must be one of the datasets passed to `train`.
        algorithm_device: the device used to run the ensemble algorithm and compute
            `algorithm_metric`.
    """

    algorithm: KWArgs
    online: bool
    prediction_transform: PredictionTransform | None
    early_stopping_patience: int | None
    algorithm_metric: MetricFn
    algorithm_metric_direction: MetricDirection
    algorithm_dataset: DatasetName
    algorithm_device: torch.device

    def __post_init__(self) -> None:
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 0
        ):
            raise ValueError('The early stopping patience must be non-negative')


EnsembleAlgorithmType: TypeAlias = Callable
"""An ensemble algorithm.

The callable must accept at least the following arguments (plus any number of other
arguments):

- `predictions: Tensor` as the first positional argument, with candidate checkpoints
  along the first dimension.
- `score_fn: Callable[[Tensor], Tensor]`, returning one higher-is-better score per
  prediction in the input pack.

Return a one-dimensional tensor of unique selected indices, or `(indices, weights)`
with one weight per selected index. Weights may be `None` for equal weighting.
"""


def make_ensemble_algorithm(
    type: EnsembleAlgorithmType,
    *,
    y_true: Tensor,
    metric_fn: MetricFn,
    metric_direction: MetricDirection,
    device: torch.device,
    **kwargs,
) -> EnsembleAlgorithmFn:
    fn = type
    del type

    y_true = y_true.to(device)

    def score_fn(y_pred: Tensor) -> Tensor:
        value = metric_fn(y_true=y_true, y_pred=y_pred)
        if value.shape != (y_pred.shape[tabpackage.nn.PACK_DIM],):
            raise ValueError(
                'The metric used to select ensemble members must return one scalar per'
                ' candidate'
            )
        return value if metric_direction == MetricDirection.MAXIMIZE else -value

    def fn_on_device(predictions: Tensor) -> tuple[Tensor, Tensor | None]:
        result = fn(predictions.to(device), score_fn=score_fn, **kwargs)
        member_idx, member_weights = (
            (result, None) if isinstance(result, Tensor) else result
        )
        return (
            member_idx.to(predictions.device),
            None if member_weights is None else member_weights.to(predictions.device),
        )

    return fn_on_device


def make_ensemble_score_fn(
    *, metric_fn: MetricFn, metric_direction: MetricDirection, y_true: Tensor
) -> Callable[[Tensor], float]:
    def score_fn(y_pred: Tensor) -> float:
        value = metric_fn(y_true=y_true, y_pred=y_pred.to(y_true.device))
        if value.ndim != 0:
            raise ValueError(
                'The metric used to score an ensemble must return a scalar tensor'
            )
        value = value.item()
        return value if metric_direction == MetricDirection.MAXIMIZE else -value

    return score_fn


def evaluate_ensemble(
    *,
    ensemble: Ensemble,
    checkpoint_store: CheckpointStore,
    labels: dict[DatasetName, Tensor],
    metric_fns: dict[DatasetName, dict[MetricName, MetricFn]],
    aux_device: torch.device,
) -> dict[DatasetName, dict[MetricName, Tensor]]:
    assert ensemble.member_checkpoint_keys is not None

    metrics = {}

    for dataset_name, dataset_metric_fns in metric_fns.items():
        y_pred = compute_ensemble_predictions(
            gather_checkpoint_predictions(
                checkpoint_store, ensemble.member_checkpoint_keys, dataset_name
            ),
            ensemble.member_weights,
            ensemble.prediction_transform,
        ).to(aux_device)

        metrics[dataset_name] = {
            k: v(y_true=labels[dataset_name], y_pred=y_pred).clone()
            for k, v in dataset_metric_fns.items()
        }
        del y_pred

    return metrics


# >>> Printing


def update_best_criterion_metric(
    best: float | None, latest: float, direction: MetricDirection
) -> float:
    return (
        latest
        if best is None
        or (
            not math.isnan(latest)
            and (
                math.isnan(best)
                or (
                    latest < best
                    if direction == MetricDirection.MINIMIZE
                    else latest > best
                )
            )
        )
        else best
    )


def format_training_progress(
    *,
    elapsed_seconds: int,
    epoch: int,
    loss: float,
    n_finished_models: int,
    n_running_models: int,
    n_pending_models: int | None,
    show_best_model: bool,
    show_best_ensemble: bool,
    best_model_criterion_metric: float | None,
    best_ensemble_criterion_metric: float | None,
) -> tuple[str, str]:
    separator = '  |  '

    # Assemble headers.
    # fmt: off
    group_headers  = ['         training        ']
    column_headers = ['    time | epoch |   loss']
    if n_pending_models is None:
        group_headers.append ('      models      ')
        column_headers.append('finished | running')
    else:
        group_headers.append ('           models           ')
        column_headers.append('finished | running | pending')
    if show_best_model and show_best_ensemble:
        group_headers.append ('       best       ')
        column_headers.append('  model | ensemble')
    elif show_best_model:
        group_headers.append ('  best ')
        column_headers.append('  model')
    elif show_best_ensemble:
        group_headers.append ('  best  ')
        column_headers.append('ensemble')
    # fmt: on
    group_header = separator.join(group_headers)
    column_header = separator.join(column_headers)

    # Assemble a row.
    row = (
        f'{datetime.timedelta(seconds=elapsed_seconds)!s:0>8} | {epoch:>5} | {loss:>6.3f}'  # noqa: E501
        f'{separator}{n_finished_models:>8} | {n_running_models:>7}'
    )
    if n_pending_models is not None:
        row += f' | {n_pending_models:>7}'
    if show_best_model:
        model_score = (
            '-'
            if best_model_criterion_metric is None
            else f'{best_model_criterion_metric:.4f}'
        )
        row += f'{separator}{model_score:>7}'
    if show_best_ensemble:
        ensemble_score = (
            '-'
            if best_ensemble_criterion_metric is None
            else f'{best_ensemble_criterion_metric:.4f}'
        )
        row += f'{" | " if show_best_model else separator}{ensemble_score:>8}'

    return '\n'.join((group_header, column_header, '-' * len(column_header))), row


def _is_notebook() -> bool:
    shell = None if get_ipython is None else get_ipython()
    return shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell'


if tqdm is None:
    ProgressBar = None

else:

    class ProgressBar(tqdm.tqdm):
        def __init__(
            self,
            batches: list[Tensor],
            *,
            training_loop_step: int,
            n_steps_per_epoch: int,
            pack_size: int,
        ) -> None:
            self.pack_size = pack_size
            super().__init__(
                batches,
                file=sys.stdout if _is_notebook() else None,
                desc=f'epoch {training_loop_step // n_steps_per_epoch + 1}',
                bar_format=(
                    '{l_bar}{bar}| {n_fmt}/{total_fmt} '
                    '[{elapsed}<{remaining}, {rate_fmt}, '
                    '{model_rate} model-it/s{postfix}]'
                ),
                ncols=min(130, shutil.get_terminal_size().columns - 1),
                leave=False,
            )

        @property
        def format_dict(self) -> dict[str, Any]:
            fields = super().format_dict
            rate = fields['rate']
            if rate is None and fields['elapsed']:
                rate = (fields['n'] - fields['initial']) / fields['elapsed']

            fields['model_rate'] = (
                '?' if rate is None else f'{rate * self.pack_size:.2f}'
            )
            return fields

# >>> `train`


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TabPackConfig:
    # Models
    n_models: int
    model: KWArgs  # Must not contain per-model hyperparameters.
    prediction_transform: PredictionTransform | None

    # Training
    loss_fn: LossFn
    optimizer: KWArgs  # Must not contain per-model hyperparameters.
    batch_size: int
    max_epochs: int | None
    early_stopping_patience: int | None

    # Per-model hyperparameters
    per_model_hyperparameters: list[HyperparameterDict] | None

    # Evaluation
    eval_metrics: dict[DatasetName, dict[MetricName, MetricFn]]
    eval_criterion: EvalCriterionConfig | None
    eval_batch_size: int

    # Ensembles
    ensembles: dict[EnsembleName, EnsembleConfig]
    ensemble_eval_metrics: dict[DatasetName, dict[MetricName, MetricFn]]
    ensemble_eval_criterion: EvalCriterionConfig | None
    stop_when_all_ensembles_are_finalized: bool

    # Efficiency
    max_pack_size: int
    device: torch.device
    aux_device: torch.device
    autocast_dtype: torch.dtype | None
    compile_model: bool
    compile_model_numerical_embeddings: bool
    compile_model_first_block: bool
    compile_optimizer_muon_newton_schulz: bool

    # Printing
    print_progress: bool
    show_progress_bar: bool


def _check_autocast_dtype(dtype: torch.dtype) -> None:
    # FP16 is not supported as the AMP data type, because it requires gradient
    # scaling, which is not implemented in this codebase.
    if dtype != torch.bfloat16:
        raise ValueError('The autocast data type must be `torch.bfloat16`')


def _check_dataset(dataset: dict[str, Tensor]) -> None:
    if not dataset:
        raise ValueError('The dataset must not be empty')

    if not {'y'} < dataset.keys() <= {'y', 'x_num', 'x_cat'}:
        raise ValueError(
            'The dataset must contain labels under `y` and features under `x_num`,'
            ' `x_cat`, or both, with no other keys'
        )

    y = dataset['y']
    if y is None:
        raise TypeError('Dataset labels must not be `None`')

    if y.ndim <= 0:
        raise ValueError('Dataset labels must have a sample dimension')

    if not torch.all(torch.isfinite(y)):
        raise ValueError('Dataset labels must be finite')

    dataset_size = len(y)
    if dataset_size == 0:
        raise ValueError('The dataset must contain at least one sample')

    if not all(x.ndim > 0 and len(x) == dataset_size for x in dataset.values()):
        raise ValueError('Each dataset tensor must contain one entry per sample')

    device = next(iter(dataset.values())).device
    if not all(x.device == device for x in dataset.values()):
        raise ValueError('All tensors in a dataset must be on the same device')

    x_num = dataset.get('x_num')
    if x_num is not None:
        if x_num.ndim != 2:
            raise ValueError('Numerical features must be two-dimensional')

        if x_num.shape[1] <= 0:
            raise ValueError('Numerical features must have at least one column')

        if x_num.dtype != torch.float32:
            raise TypeError(
                'Numerical features must have the data type `torch.float32`'
            )

        if not torch.all(torch.isfinite(x_num)):
            raise ValueError('Numerical features must be finite')

    x_cat = dataset.get('x_cat')
    if x_cat is not None:
        if x_cat.ndim != 2:
            raise ValueError('Categorical features must be two-dimensional')

        if x_cat.shape[1] <= 0:
            raise ValueError('Categorical features must have at least one column')

        if x_cat.dtype != torch.int64:
            raise TypeError(
                'Categorical features must have the data type `torch.int64`'
            )

        if not torch.all(x_cat >= 0):
            raise ValueError('Categorical feature values must be non-negative')


def _check_train_args(
    config: _TabPackConfig, datasets: dict[DatasetName, dict[str, Tensor]]
) -> None:
    if not datasets:
        raise ValueError('Provide at least one dataset')

    if 'train' not in datasets:
        raise ValueError('Provide a training dataset as `datasets["train"]`')

    train_dataset = datasets['train']
    for dataset in datasets.values():
        _check_dataset(dataset)
        if dataset is not train_dataset:
            if dataset.keys() != train_dataset.keys():
                raise ValueError(
                    'All datasets must have the same keys as the training dataset'
                )
            for key, value in dataset.items():
                train_value = train_dataset[key]
                if value.shape[1:] != train_value.shape[1:]:
                    raise ValueError(
                        'Per-sample tensor shapes must match those in the training'
                        ' dataset'
                    )

                if value.dtype != train_value.dtype:
                    raise TypeError(
                        'Each dataset tensor must have the same data type as the'
                        ' corresponding training tensor'
                    )
                del key, value, train_value
        del dataset

    if config.n_models <= 0:
        raise ValueError('The number of models must be positive')

    n_numerical_features = config.model.get('n_numerical_features')
    if n_numerical_features is not None:
        if n_numerical_features > 0:
            if 'x_num' not in train_dataset:
                raise ValueError(
                    'The model is configured to use numerical features, but the'
                    ' training dataset is missing `x_num`'
                )

            if train_dataset['x_num'].shape[1] != n_numerical_features:
                raise ValueError(
                    'The number of numerical features in the training dataset must'
                    ' match `n_numerical_features`'
                )
        else:
            if 'x_num' in train_dataset:
                raise ValueError(
                    'The model is configured to use no numerical features, so the'
                    ' dataset must not contain `x_num`'
                )
    del n_numerical_features

    # Cardinalities of categorical features must be provided and are not inferred,
    # because it is tricky to do that correctly in the general case.
    if 'categorical_cardinalities' not in config.model:
        raise ValueError(
            'Provide `categorical_cardinalities` in the model configuration'
        )

    categorical_cardinalities = config.model['categorical_cardinalities']
    if categorical_cardinalities:
        if 'x_cat' not in train_dataset:
            raise ValueError(
                'The model is configured to use categorical features, but the training'
                ' dataset is missing `x_cat`'
            )
        for dataset in datasets.values():
            if dataset['x_cat'].shape[1] != len(categorical_cardinalities):
                raise ValueError(
                    'The number of categorical features in the dataset must match the'
                    ' length of `categorical_cardinalities`'
                )

            if not torch.all(
                dataset['x_cat']
                <= torch.tensor(
                    categorical_cardinalities,
                    dtype=dataset['x_cat'].dtype,
                    device=dataset['x_cat'].device,
                )[None]
            ):
                raise ValueError(
                    'Values in each categorical feature must not exceed the'
                    ' corresponding entry in `categorical_cardinalities`'
                )
    else:
        if 'x_cat' in train_dataset:
            raise ValueError(
                'The model is configured to use no categorical features, so the dataset'
                ' must not contain `x_cat`'
            )
    del train_dataset, categorical_cardinalities

    if 'd_out' not in config.model:
        raise ValueError('Provide `d_out` in the model configuration')

    if config.batch_size <= 0:
        raise ValueError('The training batch size must be positive')

    if config.max_epochs is not None and config.max_epochs <= 0:
        raise ValueError('The maximum number of epochs must be positive')

    if config.early_stopping_patience is not None:
        if config.early_stopping_patience < 0:
            raise ValueError('The early stopping patience must be non-negative')

        if config.eval_criterion is None:
            raise ValueError(
                'To use early stopping for models, provide `eval_criterion`'
            )

    for eval_metrics, eval_criterion in (
        (config.eval_metrics, config.eval_criterion),
        (config.ensemble_eval_metrics, config.ensemble_eval_criterion),
    ):
        if not eval_metrics.keys() <= datasets.keys():
            raise ValueError(
                'Evaluation metrics are configured for datasets that were not provided'
            )

        if eval_criterion is not None:
            if eval_criterion.dataset not in eval_metrics:
                raise ValueError(
                    'No evaluation metrics are configured for the dataset selected by'
                    ' the evaluation criterion'
                )

            if eval_criterion.metric not in eval_metrics[eval_criterion.dataset]:
                raise ValueError(
                    'The metric selected by the evaluation criterion is not configured'
                    ' for its dataset'
                )
            del eval_metrics, eval_criterion

    if config.eval_batch_size <= 0:
        raise ValueError('The evaluation batch size must be positive')

    if config.ensembles:
        # The ensemble names must be valid for their use in `ModuleDict`.
        moduledict = nn.ModuleDict()
        for ensemble_name in config.ensembles:
            moduledict[ensemble_name] = nn.Identity()
            del ensemble_name
        del moduledict
        for ensemble_config in config.ensembles.values():
            if ensemble_config.algorithm_dataset not in datasets:
                raise ValueError(
                    'The dataset selected for the ensemble algorithm was not provided'
                )

            if (
                ensemble_config.early_stopping_patience is not None
                and config.ensemble_eval_criterion is None
            ):
                raise ValueError(
                    'To use early stopping for ensembles, provide'
                    ' `ensemble_eval_criterion`'
                )
        del ensemble_config

    if (
        config.max_epochs is not None
        and config.per_model_hyperparameters is not None
        and 'max_epochs' in config.per_model_hyperparameters[0]
    ):
        raise ValueError('Set `max_epochs` either globally or per model, not both')

    if not (
        config.max_epochs is not None
        or (
            config.per_model_hyperparameters is not None
            and all(
                x.get('max_epochs') is not None
                for x in config.per_model_hyperparameters
            )
        )
        or config.early_stopping_patience is not None
        or (
            config.ensembles
            and config.stop_when_all_ensembles_are_finalized
            and all(
                x.online and x.early_stopping_patience is not None
                for x in config.ensembles.values()
            )
        )
    ):
        raise ValueError(
            'Training has no stopping condition. Set an epoch limit globally or for'
            ' every model, enable early stopping for models, or configure all ensembles'
            ' with `online=True` and early stopping and set'
            ' `stop_when_all_ensembles_are_finalized=True`'
        )

    if config.per_model_hyperparameters is not None:
        if len(config.per_model_hyperparameters) != config.n_models:
            raise ValueError('Provide one hyperparameter dictionary per model')

        if not config.per_model_hyperparameters[0].keys() <= {
            'model',
            'optimizer',
            'max_epochs',
        }:
            raise ValueError(
                'Each dictionary in `per_model_hyperparameters` may contain only the'
                ' keys `model`, `optimizer`, and `max_epochs`'
            )
        # The following call ensures that all per-model hyperparameters have the
        # same structure. This is critical to do, because if an earlier model
        # specified some hyperparameter that a later model omitted, then the later
        # model would inherit the earlier model's hyperparameter value.
        transpose_dicts_recursively(config.per_model_hyperparameters)

    if config.max_pack_size <= 0 or config.max_pack_size > config.n_models:
        raise ValueError(
            'The maximum pack size must be between 1 and the number of models'
        )

    if config.autocast_dtype is not None:
        _check_autocast_dtype(config.autocast_dtype)

    if config.compile_model and (
        config.compile_model_numerical_embeddings or config.compile_model_first_block
    ):
        raise ValueError(
            'Incorrect combination of `compile_*` arguments. Choose to compile either'
            ' the full model or some parts of it, but not both.'
        )

    if config.compile_model and config.device.type != 'cuda':
        # As of PyTorch 2.14.1, the full compilation results in incorrect computations
        # on MPS. The relevant fix: https://github.com/pytorch/pytorch/pull/201460
        # Once the fix becomes available (supposedly in PyTorch 2.16), reevaluate
        # the full compilation on MPS.
        raise ValueError('The full model compilation is available only on CUDA')

    # The version lower bound is motivated only by MPS limitations, but is applied
    # to all backends just in case; this may be relaxed in future.
    if (
        config.compile_model
        or config.compile_model_numerical_embeddings
        or config.compile_model_first_block
        or config.compile_optimizer_muon_newton_schulz
    ) and torch.torch_version.TorchVersion(torch.__version__) < '2.12.0':
        raise RuntimeError('Compilation requires PyTorch 2.12.0 or newer')

    if (
        config.compile_model_numerical_embeddings
        and 'numerical_embeddings' not in config.model
    ):
        raise ValueError(
            'Cannot compile numerical embeddings because none are configured'
        )

    if config.compile_optimizer_muon_newton_schulz and not issubclass(
        config.optimizer['type'], tabpackage.optim.MuonAdamWPack
    ):
        raise ValueError(
            'The optimizer does not support compiling the Newton-Schulz iteration for'
            ' Muon'
        )

    if config.show_progress_bar and tqdm is None:
        raise ImportError('`show_progress_bar=True` requires `tqdm` to be installed')


def _infer_model_shape_kwargs(
    config: _TabPackConfig, datasets: dict[DatasetName, dict[str, Tensor]]
) -> KWArgs:
    inferred_model_kwargs = {}

    if 'n_numerical_features' not in config.model:
        inferred_model_kwargs['n_numerical_features'] = (
            datasets['train']['x_num'].shape[-1] if 'x_num' in datasets['train'] else 0
        )

    if (
        'categorical_cardinalities' not in config.model
        and 'x_cat' not in datasets['train']
    ):
        inferred_model_kwargs['categorical_cardinalities'] = []

    if (
        config.per_model_hyperparameters is not None
        and 'model' in config.per_model_hyperparameters[0]
    ):
        if (
            'max_n_blocks' not in config.model
            and 'n_blocks' in config.per_model_hyperparameters[0]['model']
        ):
            inferred_model_kwargs['max_n_blocks'] = max(
                x['model']['n_blocks'] for x in config.per_model_hyperparameters
            )
        if (
            'max_d_block' not in config.model
            and 'd_block' in config.per_model_hyperparameters[0]['model']
        ):
            inferred_model_kwargs['max_d_block'] = max(
                x['model']['d_block'] for x in config.per_model_hyperparameters
            )
        if (
            config.model.get('numerical_embeddings') is not None
            and 'max_d_embedding' not in config.model['numerical_embeddings']
            and (
                'd_embedding'
                in config.per_model_hyperparameters[0]['model'].get(
                    'numerical_embeddings', {}
                )
            )
        ):
            inferred_model_kwargs['numerical_embeddings'] = {
                'max_d_embedding': max(
                    x['model']['numerical_embeddings']['d_embedding']
                    for x in config.per_model_hyperparameters
                )
            }

    return inferred_model_kwargs


class TabPackPredictor(nn.Module):
    """An inference-only predictor for trained models and their ensembles.

    Create it with `TrainedTabPack.make_predictor` to use the selected checkpoints
    and prediction transformations from a training run.
    """

    class EnsembleHead(nn.Module):
        """Select and aggregate an ensemble's member predictions."""

        member_pack_idx: Tensor
        member_weights: Tensor | None

        def __init__(
            self,
            *,
            member_pack_idx: Tensor,
            member_weights: Tensor | None,
            prediction_transform: PredictionTransform | None,
        ) -> None:
            check_ensemble_member_idx(member_pack_idx)
            if not torch.all(member_pack_idx >= 0):
                raise ValueError('Ensemble member indices must be non-negative')

            if len(torch.unique(member_pack_idx)) != len(member_pack_idx):
                raise ValueError('Ensemble member indices must be unique')

            if member_weights is not None:
                check_ensemble_member_weights(member_weights)

                if not torch.all(torch.isfinite(member_weights)):
                    raise ValueError('Ensemble member weights must be finite')

                if not torch.all(member_weights >= 0):
                    raise ValueError('Ensemble member weights must be non-negative')

                if not torch.any(member_weights > 0):
                    raise ValueError(
                        'At least one ensemble member weight must be positive'
                    )

            super().__init__()
            self.register_buffer('member_pack_idx', member_pack_idx)
            self.register_buffer('member_weights', member_weights)
            self.prediction_transform = prediction_transform

        def forward(self, x: Tensor) -> Tensor:
            return compute_ensemble_predictions(
                x[self.member_pack_idx], self.member_weights, self.prediction_transform
            )

    def __init__(
        self,
        *,
        model_pack: tabpackage.nn.TabPack,
        autocast_dtype: torch.dtype | None,
        prediction_transform: PredictionTransform | None,
        ensembles: dict[EnsembleName, EnsembleHead],
    ) -> None:
        super().__init__()
        self.model_pack: Final = model_pack
        self.prediction_transform: Final = prediction_transform
        self.autocast_dtype: Final = autocast_dtype
        self.ensembles: Final = nn.ModuleDict(ensembles)

        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> Self:
        if mode:
            raise RuntimeError('The module supports only evaluation mode')

        return super().train(mode)

    def forward(
        self, x_num: Tensor | None, x_cat: Tensor | None
    ) -> tuple[Tensor, dict[EnsembleName, Tensor]]:
        """Make individual and ensemble predictions for a batch of new examples.

        Args:
            x_num: the numerical features.
            x_cat: the categorical features.

        Returns:
            The individual model predictions and the ensemble predictions.
        """
        if self.training:
            raise RuntimeError('The module must be in evaluation mode')

        batch = {}
        if x_num is not None:
            batch['x_num'] = x_num
        if x_cat is not None:
            batch['x_cat'] = x_cat

        y_pred_pack = eval_forward(
            self.model_pack,
            batch,
            autocast=(
                None
                if self.autocast_dtype is None
                else torch.autocast(
                    next(self.model_pack.parameters()).device.type, self.autocast_dtype
                )
            ),
            prediction_transform=self.prediction_transform,
        )
        y_pred_ensembles = {k: v(y_pred_pack) for k, v in self.ensembles.items()}

        return y_pred_pack, y_pred_ensembles


class TrainedTabPack:
    """Trained TabPack for making predictions and inspecting training results.

    Use `make_predictor` to construct prediction modules. Use other methods to access
    training and evaluation artifacts and statistics of models and ensembles.

    **Conventions**

    - Model steps, state dictionaries, and metrics come from each model's best
      checkpoint according to the evaluation criterion, or its latest evaluated
      checkpoint if no criterion was provided.
    - Ensemble artifacts use the best ensemble according to its evaluation criterion,
      or its latest update if no criterion was provided. Its member checkpoints may
      differ from the models' individual best checkpoints.
    - Checkpoint state dictionaries and evaluation metrics are stored on `aux_device`
      from `train`.
    - Returned training artifacts are deeply copied, so they can be modified as needed.
    - Checkpoint selections preserve request order and repetitions; missing checkpoint
      keys are rejected.
    - Returned modules have the shape upper bounds of the original model pack and can
      exceed the shape of the actual largest pack member.
    """

    def __init__(
        self,
        *,
        config: _TabPackConfig,
        n_steps_per_epoch: int,
        training_loop_time: float,
        training_loop_step: int,
        prediction_transform: PredictionTransform | None,
        common_model_kwargs: dict[str, Any],
        per_model_hyperparameter_store: dict[ModelID, HyperparameterDict] | None,
        history: list[dict[str, Any]],
        checkpoints: ModelCheckpoints,
        finished_best_checkpoint_keys: ModelCheckpointKeys,
        unfinished_best_checkpoint_keys: ModelCheckpointKeys,
        ensemble_prediction_transforms: dict[EnsembleName, PredictionTransform | None],
        ensemble_history: dict[EnsembleName, list[dict[str, Any]]],
    ) -> None:
        assert n_steps_per_epoch > 0
        assert training_loop_step > 0
        assert common_model_kwargs
        assert finished_best_checkpoint_keys or unfinished_best_checkpoint_keys

        self._config: Final = config
        self._n_steps_per_epoch: Final = n_steps_per_epoch
        self._training_loop_time = training_loop_time
        self._training_loop_step: Final = training_loop_step
        self._prediction_transform: Final = prediction_transform
        self._common_model_kwargs: Final = common_model_kwargs
        self._per_model_hyperparameter_store: Final = per_model_hyperparameter_store
        self._checkpoints: Final = checkpoints
        self._finished_best_checkpoint_keys: Final = finished_best_checkpoint_keys
        self._unfinished_best_checkpoint_keys: Final = unfinished_best_checkpoint_keys
        self._history: Final = history
        self._ensemble_prediction_transforms: Final = ensemble_prediction_transforms
        self._ensemble_history: Final = ensemble_history

        self._model_best_steps = {}
        for checkpoint_keys in (
            self._finished_best_checkpoint_keys,
            self._unfinished_best_checkpoint_keys,
        ):
            assert len(set(checkpoint_keys.tolist())) == len(checkpoint_keys)
            self._model_best_steps.update(
                (x.id, x.step) for x in checkpoint_keys.tolist()
            )

        self._model_checkpoint_indices = {
            key: index for index, key in enumerate(self._checkpoints.keys.tolist())
        }

    # >>> Training loop

    def get_n_steps_per_epoch(self) -> int:
        """Get the number of training steps per epoch."""
        return self._n_steps_per_epoch

    def get_training_loop_time(self) -> float:
        """Get the time spent in the training loop, in seconds."""
        return self._training_loop_time

    def get_training_loop_step(self) -> int:
        """Get the number of training steps performed by the training loop."""
        return self._training_loop_step

    # >>> Hyperparameters

    def load_common_model_kwargs(self) -> KWArgs:
        """Get common constructor arguments for the `tabpackage.nn.TabPack` module.

        The returned arguments include the ones from the originally provided `model`
        dictionary and the ones inferred before the training, in particular shape
        bounds.
        """
        return copy.deepcopy(self._common_model_kwargs)

    def load_per_model_hyperparameters(
        self, model_ids: list[ModelID]
    ) -> list[HyperparameterDict] | None:
        """Get a list of per-model hyperparameter dictionaries."""
        return (
            None
            if self._per_model_hyperparameter_store is None
            else [
                copy.deepcopy(self._per_model_hyperparameter_store[x])
                for x in model_ids
            ]
        )

    # >>> Models

    def _get_model_checkpoint_indices(
        self, *, model_ids: Tensor, model_steps: Tensor
    ) -> Tensor:
        keys = ModelCheckpointKeys(ids=model_ids, steps=model_steps)
        return torch.tensor(
            [self._model_checkpoint_indices[key] for key in keys.tolist()],
            dtype=torch.int64,
            device=_IDS_DEVICE,
        )

    def load_model_prediction_transform(
        self,
    ) -> PredictionTransform | None:
        """Load the transformation applied to the raw per-model outputs on inference."""
        return copy.deepcopy(self._prediction_transform)

    def load_finished_model_ids(self) -> Tensor:
        """Load IDs of the finished models."""
        return self._finished_best_checkpoint_keys.ids.clone()

    def load_unfinished_model_ids(self) -> Tensor:
        """Load IDs of unfinished models with evaluated checkpoints.

        The unfinished models are the ones that did not finish their training
        (because the whole TabPack's training loop stopped for some reason) and
        were evaluated at least once.
        """
        return self._unfinished_best_checkpoint_keys.ids.clone()

    def load_model_steps(self, model_ids: Tensor) -> Tensor:
        """Load the training steps of the models' checkpoints."""
        return torch.tensor(
            [self._model_best_steps[x] for x in model_ids.tolist()],
            dtype=torch.int64,
            device=_IDS_DEVICE,
        )

    def _load_model_state_dicts(
        self, *, model_ids: Tensor, model_steps: Tensor
    ) -> dict[str, Tensor]:
        return tabpackage.utils.torch.index(
            self._checkpoints.state_dicts,
            self._get_model_checkpoint_indices(
                model_ids=model_ids, model_steps=model_steps
            ),
        )

    def load_model_state_dicts(self, model_ids: Tensor) -> dict[str, Tensor]:
        """Load state dictionaries of models."""
        return self._load_model_state_dicts(
            model_ids=model_ids, model_steps=self.load_model_steps(model_ids)
        )

    def _load_model_metrics(
        self, *, model_ids: Tensor, model_steps: Tensor
    ) -> dict[DatasetName, dict[MetricName, Tensor]]:
        return tabpackage.utils.torch.index(
            self._checkpoints.metrics,
            self._get_model_checkpoint_indices(
                model_ids=model_ids, model_steps=model_steps
            ),
        )

    def load_model_metrics(
        self, model_ids: Tensor
    ) -> dict[DatasetName, dict[MetricName, Tensor]]:
        """Load metrics of models."""
        return self._load_model_metrics(
            model_ids=model_ids, model_steps=self.load_model_steps(model_ids)
        )

    def load_model_history(self) -> list[dict[str, Any]]:
        """Load the models' evaluation results recorded after each training epoch."""
        return copy.deepcopy(self._history)

    # >>> Ensembles

    def get_ensemble_names(self) -> list[EnsembleName]:
        """Get the names of the built ensembles."""
        return [k for k, v in self._ensemble_history.items() if v]

    def load_ensemble_prediction_transform(
        self, ensemble_name: EnsembleName
    ) -> PredictionTransform | None:
        """Load the transformation applied to the aggregated prediction of ensemble members on inference."""  # noqa: E501
        return copy.deepcopy(self._ensemble_prediction_transforms[ensemble_name])

    def load_ensemble_member_ids(self, ensemble_name: EnsembleName) -> Tensor:
        """Load model IDs of ensemble members."""
        return self._ensemble_history[ensemble_name][-1]['members']['ids'].clone()

    def load_ensemble_member_steps(self, ensemble_name: EnsembleName) -> Tensor:
        """Load checkpoint steps of ensemble members."""
        return self._ensemble_history[ensemble_name][-1]['members']['steps'].clone()

    def load_ensemble_member_weights(
        self, ensemble_name: EnsembleName
    ) -> Tensor | None:
        """Load weights (importances) of ensemble members."""
        weights = self._ensemble_history[ensemble_name][-1]['members'].get('weights')
        return None if weights is None else weights.clone()

    def load_ensemble_member_metrics(
        self, ensemble_name: EnsembleName
    ) -> dict[DatasetName, dict[MetricName, Tensor]]:
        """Load metrics of ensemble members."""
        return self._load_model_metrics(
            model_ids=self.load_ensemble_member_ids(ensemble_name),
            model_steps=self.load_ensemble_member_steps(ensemble_name),
        )

    def load_ensemble_metrics(
        self, ensemble_name: EnsembleName
    ) -> dict[DatasetName, dict[MetricName, Tensor]]:
        """Load metrics of an ensemble."""
        return copy.deepcopy(self._ensemble_history[ensemble_name][-1]['metrics'])

    def load_ensemble_history(
        self, ensemble_name: EnsembleName
    ) -> list[dict[str, Any]]:
        """Load ensemble's members and evaluation results at each accepted update."""
        return copy.deepcopy(self._ensemble_history[ensemble_name])

    # >>> Module factories

    def _make_model_pack(
        self, *, model_ids: Tensor, model_steps: Tensor
    ) -> tabpackage.nn.TabPack:
        """Build a model pack from model checkpoints."""
        assert len(model_ids) > 0

        model_kwargs = self.load_common_model_kwargs()
        per_model_hyperparameters = self.load_per_model_hyperparameters(
            model_ids.tolist()
        )
        if per_model_hyperparameters is not None:
            merge_dicts_recursively(
                model_kwargs,
                transpose_dicts_recursively(per_model_hyperparameters).get('model', {}),
            )

        state_dict = self._load_model_state_dicts(
            model_ids=model_ids, model_steps=model_steps
        )
        some_state_tensor = next(
            x for x in state_dict.values() if x.is_floating_point()
        )

        # Prevent parameter initialization from changing the global RNG state.
        with torch.random.fork_rng(devices=[]):
            model_pack = tabpackage.nn.TabPack(pack_size=len(model_ids), **model_kwargs)
        model_pack.to(some_state_tensor)
        model_pack.load_state_dict(state_dict)

        return model_pack

    def make_predictor(
        self,
        *,
        ensemble_names: list[EnsembleName],
        autocast_dtype: torch.dtype | None = None,
    ) -> TabPackPredictor:
        """Build a predictor for making predictions with selected ensembles.

        The predictor includes only the checkpoints needed by the requested ensembles.
        Checkpoints shared by multiple ensembles are loaded and only once, and their
        predictions are also reused between the ensembles.

        Args:
            ensemble_names: the ensembles to include. Choose at least one name from
                `get_ensemble_names()`.
            autocast_dtype: the data type used for automatic mixed precision during
                inference. Only BF16 is supported. `None` disables autocast, regardless
                of the setting used for training.
        """
        if not ensemble_names:
            raise ValueError('Select at least one ensemble')

        if not set(ensemble_names) <= set(self.get_ensemble_names()):
            raise ValueError(
                'The requested ensembles are not all available in the training'
                ' artifacts'
            )

        if autocast_dtype is not None:
            _check_autocast_dtype(autocast_dtype)

        model_ids = []
        model_steps = []
        checkpoint_keys_to_pack_indices = {}
        ensembles = {}

        for ensemble_name in ensemble_names:
            ensemble_record = self._ensemble_history[ensemble_name][-1]

            ensemble_member_pack_indices = []
            member_keys = ModelCheckpointKeys(
                ids=ensemble_record['members']['ids'],
                steps=ensemble_record['members']['steps'],
            )
            for key in member_keys.tolist():
                if key in checkpoint_keys_to_pack_indices:
                    index = checkpoint_keys_to_pack_indices[key]
                else:
                    index = len(checkpoint_keys_to_pack_indices)
                    model_ids.append(key.id)
                    model_steps.append(key.step)
                    checkpoint_keys_to_pack_indices[key] = index
                ensemble_member_pack_indices.append(index)

            ensembles[ensemble_name] = TabPackPredictor.EnsembleHead(
                member_pack_idx=torch.tensor(ensemble_member_pack_indices),
                member_weights=self.load_ensemble_member_weights(ensemble_name),
                prediction_transform=self.load_ensemble_prediction_transform(
                    ensemble_name
                ),
            )

        assert model_ids

        model_pack = self._make_model_pack(
            model_ids=torch.tensor(model_ids, device=_IDS_DEVICE),
            model_steps=torch.tensor(model_steps, device=_IDS_DEVICE),
        )
        device = next(model_pack.parameters()).device
        return TabPackPredictor(
            model_pack=model_pack,
            autocast_dtype=autocast_dtype,
            prediction_transform=self.load_model_prediction_transform(),
            ensembles=ensembles,
        ).to(device)


class RetrainedTabPack:
    """Retrained TabPack with exactly one ensemble."""

    def __init__(
        self,
        *,
        tabpack: TrainedTabPack,
        ensemble_name: EnsembleName,
        member_weights: Tensor | None,
        prediction_transform: PredictionTransform | None,
    ) -> None:
        self._tabpack = tabpack
        self._ensemble_name = ensemble_name
        self._member_weights = member_weights
        self._prediction_transform = prediction_transform

    def make_predictor(
        self, *, autocast_dtype: torch.dtype | None = None
    ) -> TabPackPredictor:
        """Build a predictor for the retrained ensemble.

        Args:
            autocast_dtype: the data type used for automatic mixed precision during
                inference. Only BF16 is supported.
        """
        if autocast_dtype is not None:
            _check_autocast_dtype(autocast_dtype)

        # Model IDs follow submission order, while checkpoints follow completion order.
        model_ids = self._tabpack.load_finished_model_ids().sort().values
        model_pack = self._tabpack._make_model_pack(
            model_ids=model_ids,
            model_steps=self._tabpack.load_model_steps(model_ids),
        )
        device = next(model_pack.parameters()).device
        return TabPackPredictor(
            model_pack=model_pack,
            autocast_dtype=autocast_dtype,
            prediction_transform=self._tabpack.load_model_prediction_transform(),
            ensembles={
                self._ensemble_name: TabPackPredictor.EnsembleHead(
                    member_pack_idx=torch.arange(len(model_ids)),
                    member_weights=(
                        None
                        if self._member_weights is None
                        else self._member_weights.clone()
                    ),
                    prediction_transform=copy.deepcopy(self._prediction_transform),
                )
            },
        ).to(device)


def _train(
    config: _TabPackConfig, datasets: dict[DatasetName, dict[str, Tensor]]
) -> TrainedTabPack:
    _check_train_args(config, datasets)

    # >>> Efficiency

    train_dataset_device = next(iter(datasets['train'].values())).device
    autocast = (
        None
        if config.autocast_dtype is None
        else torch.autocast(config.device.type, config.autocast_dtype)
    )
    compilation_context = (
        # Backward runs outside autocast; this also covers lazy compilation.
        torch._functorch.config.patch(backward_pass_autocast='off')
        if (
            config.compile_model
            or config.compile_model_numerical_embeddings
            or config.compile_model_first_block
        )
        else contextlib.nullcontext()
    )

    labels = {k: v['y'].to(config.aux_device, copy=True) for k, v in datasets.items()}

    # >>> Model IDs and hyperparameters

    _get_next_model_id = itertools.count().__next__
    if config.per_model_hyperparameters is None:
        per_model_hyperparameter_store = None
        _get_next_model_hyperparameters = dict
    else:
        per_model_hyperparameter_store = {}
        _get_next_model_hyperparameters = iter(
            config.per_model_hyperparameters
        ).__next__

    def get_next_models(n: int) -> tuple[list[ModelID], list[HyperparameterDict]]:
        ids_and_hyperparameters = [
            (_get_next_model_id(), _get_next_model_hyperparameters()) for _ in range(n)
        ]
        if per_model_hyperparameter_store is not None:
            per_model_hyperparameter_store.update(
                copy.deepcopy(ids_and_hyperparameters)
            )
        return (
            [x[0] for x in ids_and_hyperparameters],
            [x[1] for x in ids_and_hyperparameters],
        )

    initial_pack_size = config.max_pack_size
    initial_model_ids, initial_per_model_hyperparameters = get_next_models(
        initial_pack_size
    )
    initial_per_model_hyperparameters_T = transpose_dicts_recursively(
        initial_per_model_hyperparameters
    )
    del initial_per_model_hyperparameters

    # >>> Model pack

    model_kwargs = copy.deepcopy(config.model)
    merge_dicts_recursively(model_kwargs, _infer_model_shape_kwargs(config, datasets))
    # Preserve the common arguments for materializing ensembles.
    common_model_kwargs = copy.deepcopy(model_kwargs)
    # Merge the hyperparameters of the initial models.
    merge_dicts_recursively(
        model_kwargs, initial_per_model_hyperparameters_T.pop('model', {})
    )

    model_pack = tabpackage.nn.TabPack(pack_size=initial_pack_size, **model_kwargs)
    model_pack.to(config.device)

    if config.compile_model:
        model_pack.forward = torch.compile(  # ty: ignore[invalid-assignment]
            model_pack.forward, options={'emulate_precision_casts': True}
        )
    if config.compile_model_numerical_embeddings and model_pack.num_module is not None:
        model_pack.num_module.forward = torch.compile(model_pack.num_module.forward)
    if config.compile_model_first_block:
        block = model_pack.backbone.blocks[0]
        block.forward = torch.compile(
            block.forward,
            # Emulate eager precision casts to reduce numerical differences from
            # the non-compiled variant.
            options={'emulate_precision_casts': True},
        )

    # >>> Training

    support_replacement = config.max_pack_size < config.n_models

    optimizer_kwargs = copy.deepcopy(config.optimizer)
    merge_dicts_recursively(
        optimizer_kwargs, initial_per_model_hyperparameters_T.pop('optimizer', {})
    )
    if config.compile_optimizer_muon_newton_schulz:
        optimizer_kwargs['compile_muon_newton_schulz'] = True
    optimizer_pack = make_optimizer_pack(
        model_pack, support_replacement=support_replacement, **optimizer_kwargs
    )
    del optimizer_kwargs

    train_size = len(datasets['train']['y'])
    n_steps_per_epoch = math.ceil(train_size / config.batch_size)

    initial_max_epochs = initial_per_model_hyperparameters_T.pop(
        'max_epochs', config.max_epochs
    )
    if not isinstance(initial_max_epochs, list):
        initial_max_epochs = [initial_max_epochs] * initial_pack_size

    # >>> Evaluation

    eval_batch_size = config.eval_batch_size
    history: list[dict[str, Any]] = []

    # >>> State

    training_loop_step = 0
    training_state = TrainingStatePack(
        ids=torch.tensor(initial_model_ids, device=_IDS_DEVICE),
        max_steps=torch.tensor(
            [
                _max_epochs_to_max_steps(x, n_steps_per_epoch=n_steps_per_epoch)
                for x in initial_max_epochs
            ],
            dtype=torch.int64,
            device=_IDS_DEVICE,
        ),
        steps=torch.zeros(initial_pack_size, dtype=torch.int64, device=_IDS_DEVICE),
        n_consecutive_bad_updates=torch.zeros(
            initial_pack_size, dtype=torch.int64, device=_IDS_DEVICE
        ),
    )
    # During training, running models' best checkpoints live in the training state.
    # The following store, in turn, keeps finished models' best checkpoints and
    # checkpoints selected by ensembles, which may come from earlier training steps. It
    # _must not_ contain duplicates. To maintain this invariant, use the specialized
    # function for rebuilding the store with new checkpoints.
    checkpoint_store: CheckpointStore = []
    finished_best_checkpoint_keys = ModelCheckpointKeys(
        ids=torch.empty(0, dtype=torch.int64, device=_IDS_DEVICE),
        steps=torch.empty(0, dtype=torch.int64, device=_IDS_DEVICE),
    )

    # Ensure that all hyperparameters have been used.
    assert not initial_per_model_hyperparameters_T
    del initial_pack_size, initial_model_ids, initial_per_model_hyperparameters_T

    # >>> Ensembles

    ensembles = {
        k: Ensemble(
            algorithm_fn=make_ensemble_algorithm(
                y_true=labels[v.algorithm_dataset],
                metric_fn=v.algorithm_metric,
                metric_direction=v.algorithm_metric_direction,
                device=v.algorithm_device,
                **v.algorithm,
            ),
            algorithm_dataset=v.algorithm_dataset,
            online=v.online,
            prediction_transform=v.prediction_transform,
            score_fn=(
                None
                if config.ensemble_eval_criterion is None
                else make_ensemble_score_fn(
                    metric_fn=config.ensemble_eval_metrics[
                        config.ensemble_eval_criterion.dataset
                    ][config.ensemble_eval_criterion.metric],
                    metric_direction=config.ensemble_eval_criterion.metric_direction,
                    y_true=labels[config.ensemble_eval_criterion.dataset],
                )
            ),
            score_dataset=(
                None
                if config.ensemble_eval_criterion is None
                else config.ensemble_eval_criterion.dataset
            ),
            early_stopping_patience=v.early_stopping_patience,
        )
        for k, v in config.ensembles.items()
    }
    ensemble_history: dict[EnsembleName, list[dict[str, Any]]] = {
        name: [] for name in ensembles
    }

    def get_prediction_dataset_names() -> tuple[DatasetName, ...]:
        """Get datasets for which predictions should be stored in model checkpoints.

        The important property of this function is that its output can only shrink over
        time, but never grow.
        """
        dataset_names = [
            x.algorithm_dataset for x in ensembles.values() if x.is_running()
        ]
        return (
            # Return the datasets used for building and evaluating running ensembles.
            tuple(dict.fromkeys([*dataset_names, *config.ensemble_eval_metrics]))
            if dataset_names
            else ()
        )

    prediction_dataset_names = get_prediction_dataset_names()

    # >>> Printing

    best_model_criterion_metric = None
    best_ensemble_criterion_metric = None

    # >>> Training loop

    # Most training logic is kept inline to make the loop easy to copy and customize.
    # Alternatively, local closures could make this specific loop more readable.

    training_loop_start_time = time.perf_counter()

    def get_training_loop_time() -> float:
        return time.perf_counter() - training_loop_start_time

    training_loop_is_running = True
    while training_loop_is_running:
        # >>> Training phase

        model_pack.train()

        batch_losses = []
        batch_sizes = []
        batch_idx_list = make_shuffled_batch_indices_pack(
            total_size=train_size,
            batch_size=config.batch_size,
            pack_size=training_state.pack_size,
            device=train_dataset_device,
        )

        for batch_idx in (
            ProgressBar(
                batch_idx_list,
                training_loop_step=training_loop_step,
                n_steps_per_epoch=n_steps_per_epoch,
                pack_size=training_state.pack_size,
            )
            if config.show_progress_bar and ProgressBar is not None
            else batch_idx_list
        ):  # batch_idx.shape: (pack_size, batch_size)
            batch = {
                k: v[batch_idx].to(config.device) for k, v in datasets['train'].items()
            }

            with compilation_context:
                y_pred = forward(model_pack, batch, autocast=autocast)
                losses = config.loss_fn(y_pred, batch['y'])
                if losses.shape != (model_pack.pack_size,):
                    raise ValueError(
                        'The loss function must return one scalar per model'
                    )

                # The scale of the gradients should not depend on the number of models,
                # so the individual losses are summed, not averaged.
                loss = losses.sum()
                loss.backward()
            optimizer_pack.step()
            optimizer_pack.zero_grad()

            training_loop_step += 1
            training_state.steps.add_(1)

            batch_losses.append(loss.detach())
            batch_sizes.append(batch_idx.shape[tabpackage.nn.BATCH_DIM])

            del batch, batch_idx, losses, loss
        del batch_idx_list

        # >>> Evaluation phase

        # Evaluate the models.
        (latest_predictions, latest_metrics), eval_batch_size = evaluate(
            model_pack=model_pack,
            prediction_transform=config.prediction_transform,
            datasets=datasets,
            labels=labels,
            metric_fns=config.eval_metrics,
            prediction_dataset_names=prediction_dataset_names,
            batch_size=eval_batch_size,
            device=config.device,
            aux_device=config.aux_device,
            autocast=autocast,
        )
        history.append(
            {
                'time': get_training_loop_time(),
                'training_loop_step': training_loop_step,
                'ids': training_state.ids.clone(),
                'steps': training_state.steps.clone(),
                'metrics': latest_metrics,
            }
        )
        latest_checkpoints_view = ModelCheckpoints(
            keys=ModelCheckpointKeys(
                ids=training_state.ids, steps=training_state.steps
            ),
            state_dicts=model_pack.state_dict(),
            predictions=latest_predictions,
            metrics=latest_metrics,
        )

        # Update the best criterion metric.
        if config.print_progress and config.eval_criterion is not None:
            for value in latest_metrics[config.eval_criterion.dataset][
                config.eval_criterion.metric
            ].tolist():
                best_model_criterion_metric = update_best_criterion_metric(
                    best_model_criterion_metric,
                    value,
                    config.eval_criterion.metric_direction,
                )
                del value

        del latest_predictions, latest_metrics

        # Update the best checkpoints.
        update_training_state(
            training_state,
            latest_checkpoints=latest_checkpoints_view,
            eval_criterion=config.eval_criterion,
        )
        assert training_state.best_checkpoints is not None

        # Determine which models should continue running and which should be stopped.
        running_pack_mask = training_state.is_running(
            early_stopping_patience=config.early_stopping_patience
        )
        running_pack_idx = torch.nonzero(running_pack_mask)[:, 0]
        stopped_pack_idx = torch.nonzero(~running_pack_mask)[:, 0]

        # Register the best checkpoint keys of the stopped models.
        finished_best_checkpoint_keys = tabpackage.utils.torch.cat(
            [
                finished_best_checkpoint_keys,
                tabpackage.utils.torch.index(
                    training_state.best_checkpoints.keys, stopped_pack_idx
                ),
            ]
        )

        # >>> Ensemble updates

        if any(x.is_running() for x in ensembles.values()):
            # The candidate checkpoint store can contain duplicates.
            candidate_checkpoint_store = [
                *checkpoint_store,
                training_state.best_checkpoints,
                latest_checkpoints_view,
            ]

            running_latest_checkpoint_keys = tabpackage.utils.torch.index(
                latest_checkpoints_view.keys, running_pack_idx
            )

            for ensemble_name, ensemble in ensembles.items():
                if ensemble.is_running():
                    ensemble_improved = ensemble.update(
                        checkpoint_store=candidate_checkpoint_store,
                        finished_best_checkpoint_keys=finished_best_checkpoint_keys,
                        running_latest_checkpoint_keys=running_latest_checkpoint_keys,
                    )
                    if ensemble_improved:
                        assert ensemble.member_checkpoint_keys is not None
                        ensemble_metrics = evaluate_ensemble(
                            ensemble=ensemble,
                            checkpoint_store=candidate_checkpoint_store,
                            labels=labels,
                            metric_fns=config.ensemble_eval_metrics,
                            aux_device=config.aux_device,
                        )
                        ensemble_history[ensemble_name].append(
                            {
                                'time': get_training_loop_time(),
                                'training_loop_step': training_loop_step,
                                'members': {
                                    'ids': ensemble.member_checkpoint_keys.ids.clone(),
                                    'steps': (
                                        ensemble.member_checkpoint_keys.steps.clone()
                                    ),
                                    **(
                                        {}
                                        if ensemble.member_weights is None
                                        else {'weights': ensemble.member_weights}
                                    ),
                                },
                                'metrics': ensemble_metrics,
                            }
                        )

                        if (
                            config.print_progress
                            and config.ensemble_eval_criterion is not None
                        ):
                            best_ensemble_criterion_metric = (
                                update_best_criterion_metric(
                                    best_ensemble_criterion_metric,
                                    ensemble_metrics[
                                        config.ensemble_eval_criterion.dataset
                                    ][config.ensemble_eval_criterion.metric].item(),
                                    config.ensemble_eval_criterion.metric_direction,
                                )
                            )
                        del ensemble_metrics

            del candidate_checkpoint_store, running_latest_checkpoint_keys

        # >>> Checkpoint store update

        # After the ensemble updates, the list of datasets for which predictions should
        # be stored may shrink.
        prediction_dataset_names = get_prediction_dataset_names()

        # Prune predictions for datasets that are not needed anymore. This must be done,
        # because checkpoints computed at the next epoch may also have the reduced set
        # of datasets with saved predictions, and the checkpoint structure must be the
        # same across all checkpoints.
        for checkpoints in [
            *checkpoint_store,
            training_state.best_checkpoints,
            latest_checkpoints_view,
        ]:
            for dataset_name in list(checkpoints.predictions):
                if dataset_name not in prediction_dataset_names:
                    del checkpoints.predictions[dataset_name]
            del checkpoints

        # Keep only the finished best checkpoints and the checkpoints of the
        # ensemble members.
        checkpoint_store = rebuild_checkpoint_store(
            checkpoint_store,
            [training_state.best_checkpoints, latest_checkpoints_view],
            required_keys={
                *finished_best_checkpoint_keys.tolist(),
                *itertools.chain.from_iterable(
                    x.member_checkpoint_keys.tolist()
                    for x in ensembles.values()
                    if x.member_checkpoint_keys is not None
                ),
            },
            state_dict_device=config.aux_device,
            # The best and latest checkpoints contain references to data that is updated
            # in place during training, so they must be copied.
            copy_new_checkpoints=True,
        )

        del latest_checkpoints_view

        # >>> Stopping condition

        training_loop_is_running = len(
            finished_best_checkpoint_keys
        ) < config.n_models and (
            not ensembles
            or not config.stop_when_all_ensembles_are_finalized
            or any(x.is_running() for x in ensembles.values())
        )

        # >>> Models replacement and removal

        epoch_pack_size = model_pack.pack_size
        n_finished_models = len(finished_best_checkpoint_keys)
        n_running_models = len(running_pack_idx)
        n_stopped_models = len(stopped_pack_idx)

        if training_loop_is_running:
            n_models_to_replace = min(
                n_stopped_models, config.n_models - n_finished_models - n_running_models
            )

            if n_models_to_replace > 0:
                # Replace some of the stopped models with new ones.
                assert support_replacement
                assert isinstance(optimizer_pack, tabpackage.optim.OptimizerPack)

                new_model_ids, new_per_model_hyperparameters = get_next_models(
                    n_models_to_replace
                )
                pack_replace(
                    model_pack,
                    optimizer_pack,
                    training_state,
                    pack_idx=stopped_pack_idx[:n_models_to_replace],
                    ids=new_model_ids,
                    per_model_hyperparameters=new_per_model_hyperparameters,
                    n_steps_per_epoch=n_steps_per_epoch,
                )
                del new_model_ids, new_per_model_hyperparameters

            if n_stopped_models > n_models_to_replace:
                # Remove the stopped models that were not replaced. Importantly,
                # the removal invalidates all previously computed pack indices.
                training_state = pack_remove(
                    model_pack,
                    optimizer_pack,
                    training_state,
                    pack_idx=stopped_pack_idx[n_models_to_replace:],
                )

            n_running_models += n_models_to_replace
            del n_models_to_replace
        del running_pack_mask, running_pack_idx, stopped_pack_idx

        # >>> Printing

        if config.print_progress:
            epoch = training_loop_step // n_steps_per_epoch

            epoch_mean_loss = math.fsum(
                loss * size
                for loss, size in zip(
                    torch.stack(batch_losses).tolist(), batch_sizes, strict=True
                )
            ) / math.fsum(batch_sizes)

            header, row = format_training_progress(
                elapsed_seconds=int(get_training_loop_time()),
                epoch=epoch,
                loss=epoch_mean_loss / epoch_pack_size,
                n_finished_models=n_finished_models,
                n_running_models=n_running_models,
                n_pending_models=(
                    config.n_models - n_finished_models - n_running_models
                    if support_replacement
                    else None
                ),
                show_best_model=config.eval_criterion is not None,
                show_best_ensemble=(
                    bool(config.ensembles)
                    and config.ensemble_eval_criterion is not None
                ),
                best_model_criterion_metric=best_model_criterion_metric,
                best_ensemble_criterion_metric=best_ensemble_criterion_metric,
            )
            if epoch == 1:
                print(f'\n{header}')
            print(row)
            del epoch, epoch_mean_loss, header, row

        del epoch_pack_size, n_finished_models, n_running_models, n_stopped_models

    # >>> Finish

    training_loop_time = get_training_loop_time()

    # Ensemble-level stopping can leave models unfinished; save their best
    # checkpoints too, even if no ensemble selected them.
    running_pack_mask = training_state.is_running(
        early_stopping_patience=config.early_stopping_patience
    )
    assert training_state.best_checkpoints is not None
    unfinished_best_checkpoint_keys = tabpackage.utils.torch.index(
        training_state.best_checkpoints.keys, running_pack_mask
    )
    del running_pack_mask

    checkpoint_store = rebuild_checkpoint_store(
        checkpoint_store,
        [training_state.best_checkpoints],
        required_keys={
            *finished_best_checkpoint_keys.tolist(),
            *unfinished_best_checkpoint_keys.tolist(),
            *itertools.chain.from_iterable(
                x.member_checkpoint_keys.tolist()
                for x in ensembles.values()
                if x.member_checkpoint_keys is not None
            ),
        },
        state_dict_device=config.aux_device,
        copy_new_checkpoints=False,  # The training is over, so no need to copy.
    )
    assert checkpoint_store
    for checkpoints in checkpoint_store:
        # Prune unused checkpoint data.
        checkpoints.predictions.clear()

    return TrainedTabPack(
        config=config,
        n_steps_per_epoch=n_steps_per_epoch,
        training_loop_time=training_loop_time,
        training_loop_step=training_loop_step,
        prediction_transform=config.prediction_transform,
        common_model_kwargs=common_model_kwargs,
        per_model_hyperparameter_store=per_model_hyperparameter_store,
        checkpoints=tabpackage.utils.torch.cat(checkpoint_store),
        history=history,
        finished_best_checkpoint_keys=finished_best_checkpoint_keys,
        unfinished_best_checkpoint_keys=unfinished_best_checkpoint_keys,
        ensemble_prediction_transforms={
            k: v.prediction_transform for k, v in ensembles.items()
        },
        ensemble_history=ensemble_history,
    )


# >>> `train`


@typing.final
class _Missing:
    pass


_MISSING = _Missing()


def get_next_seed(rng: random.Random) -> int:
    # The following heuristic produces seed sequences that should be good enough for the
    # intended purposes. `Random.random` is used, because it is guaranteed to be
    # reproducible between Python versions. `Random.random` produces values of the form
    # `k / 2**53`, so the multiplication by `2**53` recovers `k`.
    return int(rng.random() * (1 << 53))


@contextlib.contextmanager
def _seed_rngs(seed: int, *devices: torch.device):
    rng = random.Random(seed)

    cuda_devices = sorted(
        {
            x.index if x.index is not None else torch.cuda.current_device()
            for x in devices
            if x.type == 'cuda'
        }
    )
    use_mps = any(x.type == 'mps' for x in devices)

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=cuda_devices))
        if use_mps:
            stack.enter_context(torch.random.fork_rng(devices=[0], device_type='mps'))

        torch.default_generator.manual_seed(get_next_seed(rng))
        for cuda_device in cuda_devices:
            with torch.cuda.device(cuda_device):
                torch.cuda.manual_seed(get_next_seed(rng))
        if use_mps:
            torch.mps.manual_seed(get_next_seed(rng))

        yield


def train(
    *,
    datasets: dict[DatasetName, dict[str, Tensor]],
    n_models: int | _Missing = _MISSING,
    model: KWArgs,
    prediction_transform: PredictionTransform | None,
    loss_fn: LossFn,
    optimizer: KWArgs,
    batch_size: int,
    max_epochs: int | None,
    early_stopping_patience: int | _Missing | None = _MISSING,
    per_model_hyperparameters: list[HyperparameterDict] | None,
    eval_metrics: dict[DatasetName, dict[MetricName, MetricFn]],
    eval_criterion: EvalCriterionConfig | None,
    eval_batch_size: int = 32768,
    ensembles: dict[EnsembleName, EnsembleConfig],
    ensemble_eval_metrics: (
        dict[DatasetName, dict[MetricName, MetricFn]] | _Missing
    ) = _MISSING,
    ensemble_eval_criterion: EvalCriterionConfig | _Missing | None = _MISSING,
    stop_when_all_ensembles_are_finalized: bool | _Missing = _MISSING,
    seed: int | None = None,
    max_pack_size: int | None = None,
    device: int | str | torch.device,
    aux_device: int | str | torch.device,
    autocast_dtype: Literal['bfloat16'] | torch.dtype | None,
    compile_model: bool = False,
    compile_model_numerical_embeddings: bool = False,
    compile_model_first_block: bool = False,
    compile_optimizer_muon_newton_schulz: bool = False,
    print_progress: bool = True,
    show_progress_bar: bool | None = None,
) -> TrainedTabPack:
    """Train and ensemble many MLPs in parallel.

    Note:
        In the context of this function, the MLPs are referred to as "models".

    The function trains many models in parallel, each with its own hyperparameters, and
    can build ensembles on the fly during training. After each epoch, the models are
    evaluated, their best or latest checkpoints are saved, and the ensembles are
    updated.

    Each model stops according to its own stopping conditions. If the number of
    models trained in parallel is limited, models that have not started yet begin
    training as others finish. Depending on the provided arguments, the run continues
    either until all models finish or until all ensembles are finalized (even if some
    models are still training).

    The result contains the selected checkpoints for making predictions with individual
    models or ensembles on new data, along with evaluation histories for inspecting
    their performance.

    Args:
        datasets: the named datasets, including the required `"train"` dataset. Each
            dataset is a dictionary mapping `"y"` to labels and `"x_num"` and/or
            `"x_cat"` to numerical and categorical features, respectively.
        n_models: the total number of models to train. When omitted, it is inferred
            from the length of `per_model_hyperparameters`.
        model: the shared keyword arguments for `tabpackage.nn.TabPack`, including
            the required `d_out` and `categorical_cardinalities` arguments. Must not
            contain per-model hyperparameters (use `per_model_hyperparameters` instead).
        prediction_transform: the transformation applied to model outputs for
            evaluation and inference, such as softmax for classification models.
            `None` leaves the outputs unchanged.
        loss_fn: a callable that accepts raw predictions and labels as positional
            arguments, both with a leading model dimension. It returns one scalar
            loss per model as a tensor of shape `(pack_size,)`.
        optimizer: the optimizer configuration dictionary. The `"type"` entry
            specifies the optimizer class; the remaining entries are keyword
            arguments shared by all models. Must not contain per-model optimization
            hyperparameters (use `per_model_hyperparameters` instead).
        batch_size: the number of training examples per model per training step.
        max_epochs: the maximum number of training epochs per model. `None` means using
            per-model epoch limits or other stopping conditions.
        early_stopping_patience: the number of consecutive epochs without
            improvement in `eval_criterion` tolerated before stopping each model.
            `None` disables model-level early stopping. It must be provided when
            `eval_criterion` is set.
        per_model_hyperparameters: a list of hyperparameter dictionaries, one
            per model, with optional `"model"`, `"optimizer"`, and `"max_epochs"`
            entries. `None` means using only the shared hyperparameters, in which case
            models differ only in aspects affected by randomness, such as
            initialization, training batch sequences, dropout masks, etc.
        eval_metrics: the metric callables evaluated after each epoch, grouped by
            dataset name and metric name, returning tensors with per-model metrics. See
            `MetricFn` for details.
        eval_criterion: the evaluation criterion used to select each model's best
            checkpoint. `None` means always selecting its latest evaluated checkpoint.
        eval_batch_size: the initial number of examples per evaluation batch. May be
            automatically reduced during execution to fit the evaluation workload into
            the available device memory.
        ensembles: a dictionary mapping ensemble names to ensemble configurations.
            An empty dictionary disables ensembling.
        ensemble_eval_metrics: the metric callables for ensembles, grouped by
            dataset name and metric name. See `MetricFn` for details. Must be provided
            when `ensembles` is nonempty.
        ensemble_eval_criterion: the evaluation criterion used to select the best
            version of each ensemble. `None` means always selecting the latest update.
            Must be provided when `ensembles` is nonempty.
        stop_when_all_ensembles_are_finalized: whether to end training once all
            ensembles are finalized, even if some models are still training.
            Must be provided when `ensembles` is nonempty.
        seed: the seed for the PyTorch **global** random generators used during
            training. Also, if provided, when the function returns, the state of the
            generators is restored to what it was before the function call. `None` means
            using the current generator states without restoring their state in the end.
        max_pack_size: the maximum number of models trained in parallel. `None` means
            training all models at once.
        device: the device used for model training and evaluation.
        aux_device: the device used to compute evaluation metrics and store
            predictions and checkpoints.
        autocast_dtype: the data type used for automatic mixed precision
            during forward passes. Only BF16 is supported. `None` means not using
            autocast.
        compile_model: whether to apply `torch.compile` to the whole model.
        compile_model_numerical_embeddings: whether to apply `torch.compile` to
            the numerical embedding module's forward pass, if the model has one.
        compile_model_first_block: whether to apply `torch.compile` to the first
            backbone block's forward pass.
        compile_optimizer_muon_newton_schulz: whether to compile the Muon
            optimizer's Newton-Schulz computation.
        print_progress: whether to print a training summary after each epoch.
        show_progress_bar: whether to use `tqdm` (if installed) to show training batch
            progress within each epoch.

    Returns:
        A trained TabPack for creating predictors and accessing training and evaluation
        artifacts. See `TrainedTabPack` for details.
    """
    config_kwargs = locals()
    del config_kwargs['datasets']
    del config_kwargs['seed']

    # This function only normalizes public arguments and resolves defaults. Everything
    # else, including validation of the resulting configuration, should happen in
    # `_train`.

    if n_models is _MISSING and per_model_hyperparameters is None:
        raise ValueError(
            'Provide the number of models or the list of per-model hyperparameters to'
            ' specify how many models to train'
        )

    if n_models is _MISSING:
        assert per_model_hyperparameters is not None
        config_kwargs['n_models'] = len(per_model_hyperparameters)

    if eval_criterion is None:
        if early_stopping_patience is _MISSING:
            config_kwargs['early_stopping_patience'] = None
    elif early_stopping_patience is _MISSING:
        raise ValueError(
            'Set the early stopping patience explicitly when using an evaluation'
            ' criterion; use `None` to disable early stopping'
        )

    if ensembles:
        if ensemble_eval_metrics is _MISSING:
            raise ValueError(
                'Set `ensemble_eval_metrics` explicitly when using ensembles; use an'
                ' empty dictionary if no evaluation metrics are needed'
            )

        if ensemble_eval_criterion is _MISSING:
            raise ValueError(
                'Set `ensemble_eval_criterion` explicitly when using ensembles; use'
                ' `None` to accept ensemble updates without scoring them'
            )

        if stop_when_all_ensembles_are_finalized is _MISSING:
            raise ValueError(
                'Set `stop_when_all_ensembles_are_finalized` explicitly when using'
                ' ensembles to choose whether training should stop when all are'
                ' finalized'
            )
    else:
        if ensemble_eval_metrics is _MISSING:
            config_kwargs['ensemble_eval_metrics'] = {}
        if ensemble_eval_criterion is _MISSING:
            config_kwargs['ensemble_eval_criterion'] = None
        if stop_when_all_ensembles_are_finalized is _MISSING:
            config_kwargs['stop_when_all_ensembles_are_finalized'] = False

    if max_pack_size is None:
        config_kwargs['max_pack_size'] = config_kwargs['n_models']
    elif max_pack_size >= config_kwargs['n_models']:
        warnings.warn(
            f'The provided `max_pack_size={max_pack_size}` is no less than'
            f' the number of models {config_kwargs["n_models"]} and therefore will have'
            ' no effect. Omit `max_pack_size` to avoid this warning',
            stacklevel=2,
        )
        config_kwargs['max_pack_size'] = config_kwargs['n_models']

    if not isinstance(device, torch.device):
        config_kwargs['device'] = torch.device(device)
    if not isinstance(aux_device, torch.device):
        config_kwargs['aux_device'] = torch.device(aux_device)
    if isinstance(autocast_dtype, str):
        if autocast_dtype != 'bfloat16':
            raise ValueError(
                'When the autocast data type is a string, it must be `"bfloat16"`'
            )
        config_kwargs['autocast_dtype'] = torch.bfloat16

    if show_progress_bar is None:
        if tqdm is None:
            warnings.warn(
                'Install `tqdm` to see the per-epoch progress bar during training.'
                ' Pass `show_progress_bar=False` to avoid this warning',
                stacklevel=2,
            )
        config_kwargs['show_progress_bar'] = tqdm is not None

    config = _TabPackConfig(**config_kwargs)

    # The training uses global RNGs of following devices:
    # - CPU, for the model initialization.
    # - `device`, for training (e.g., for dropout mask generation).
    # - The training-data device, for shuffling training batches.
    with (
        contextlib.nullcontext()
        if seed is None
        else _seed_rngs(
            seed, config.device, next(iter(datasets['train'].values())).device
        )
    ):
        return _train(config, datasets)


def retrain(
    tabpack: TrainedTabPack,
    *,
    ensemble_name: EnsembleName,
    dataset: dict[str, Tensor],
    n_repeats: int = 1,
    seed: int | None = None,
) -> RetrainedTabPack:
    """Retrain one of TabPack's ensemble from scratch on full training data.

    Each member trains for the number of complete epochs represented by its checkpoint
    selected as the ensemble's member. Validation, early stopping, and ensemble
    selection are disabled. Most of other settings are reused from the original training
    run.

    Args:
        tabpack: the original trained TabPack.
        ensemble_name: the ensemble to retrain.
        dataset: the full training dataset. Redo preprocessing on the full training data
            before calling this function.
        n_repeats: the number of independently initialized copies of each ensemble
            member.
        seed: the random seed for retraining, with the same behavior as in `train`.

    Returns:
        A retrained ensemble.
    """
    if n_repeats <= 0:
        raise ValueError('The number of repeats must be positive')

    member_ids = tabpack.load_ensemble_member_ids(ensemble_name)
    member_steps = tabpack.load_ensemble_member_steps(ensemble_name)
    member_weights = tabpack.load_ensemble_member_weights(ensemble_name)
    per_model_hyperparameters = tabpack.load_per_model_hyperparameters(
        member_ids.tolist()
    )
    if per_model_hyperparameters is None:
        per_model_hyperparameters = [{} for _ in member_ids]

    for hyperparameters, steps in zip(
        per_model_hyperparameters, member_steps.tolist(), strict=True
    ):
        hyperparameters['max_epochs'] = steps // tabpack.get_n_steps_per_epoch()

    n_models = len(member_ids) * n_repeats
    config = dataclasses.replace(
        tabpack._config,
        n_models=n_models,
        max_pack_size=min(tabpack._config.max_pack_size, n_models),
        model=tabpack.load_common_model_kwargs(),
        prediction_transform=tabpack.load_model_prediction_transform(),
        per_model_hyperparameters=[
            copy.deepcopy(hyperparameters)
            for _ in range(n_repeats)
            for hyperparameters in per_model_hyperparameters
        ],
        max_epochs=None,
        early_stopping_patience=None,
        eval_metrics={},
        eval_criterion=None,
        ensembles={},
        ensemble_eval_metrics={},
        ensemble_eval_criterion=None,
        stop_when_all_ensembles_are_finalized=False,
    )
    with (
        contextlib.nullcontext()
        if seed is None
        else _seed_rngs(seed, config.device, next(iter(dataset.values())).device)
    ):
        retrained_tabpack = _train(config, {'train': dataset})

    return RetrainedTabPack(
        tabpack=retrained_tabpack,
        ensemble_name=ensemble_name,
        member_weights=(
            None
            if member_weights is None
            else member_weights.repeat(n_repeats) / n_repeats
        ),
        prediction_transform=tabpack.load_ensemble_prediction_transform(ensemble_name),
    )
