import math
import random
from functools import partial
from typing import Any, Generic, Literal, Self, TypeAlias, TypedDict, TypeVar

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition
from torch import Tensor

import tabpackage
from tabpackage import ensemble, metrics

KWArgs: TypeAlias = dict[str, Any]
DatasetName: TypeAlias = str
MetricName: TypeAlias = Literal['accuracy', 'log_loss', 'roc_auc', 'mse', 'mae', 'r2']


# >>> Per-model hyperparameters


class Distribution(BaseEstimator):
    def _get_bounds(self) -> tuple[float, float]:
        raise NotImplementedError

    def sample(self, rng: random.Random) -> float:
        """Draw a value using a given random generator."""
        raise NotImplementedError


class IntUniformDistribution(Distribution):
    """A uniform distribution over integers within inclusive bounds."""

    def __init__(self, min: int, max: int) -> None:
        self.min = min
        self.max = max

    def _get_bounds(self) -> tuple[int, int]:
        if self.min > self.max:
            raise ValueError('`min` must not exceed `max`')

        return self.min, self.max

    def sample(self, rng: random.Random) -> int:
        self._get_bounds()
        return rng.randint(self.min, self.max)


class UniformDistribution(Distribution):
    """A uniform distribution between finite bounds."""

    def __init__(self, min: float, max: float) -> None:
        self.min = min
        self.max = max

    def _get_bounds(self) -> tuple[float, float]:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError('`min` and `max` must be finite')
        if self.min > self.max:
            raise ValueError('`min` must not exceed `max`')

        return self.min, self.max

    def sample(self, rng: random.Random) -> float:
        self._get_bounds()
        return rng.uniform(self.min, self.max)


class LogUniformDistribution(UniformDistribution):
    """A log-uniform distribution between positive finite bounds."""

    def _get_bounds(self) -> tuple[float, float]:
        bounds = super()._get_bounds()
        if self.min <= 0.0:
            raise ValueError('`min` must be positive')

        return bounds

    def sample(self, rng: random.Random) -> float:
        self._get_bounds()
        return math.exp(rng.uniform(math.log(self.min), math.log(self.max)))


_DistributionT = TypeVar('_DistributionT', bound=Distribution)


class OptionalDistribution(Distribution, Generic[_DistributionT]):
    """A mixture of a given distribution (probability p) and zero."""

    def __init__(self, distribution: _DistributionT, p: float = 0.5) -> None:
        self.distribution = distribution
        self.p = p

    def _get_bounds(self) -> tuple[float, float]:
        if not math.isfinite(self.p) or self.p < 0.0 or self.p > 1.0:
            raise ValueError('`p` must be finite and between 0 and 1, inclusive')

        min_value, max_value = self.distribution._get_bounds()
        return min(0.0, min_value), max(0.0, max_value)

    def sample(self, rng: random.Random) -> float:
        self._get_bounds()
        return self.distribution.sample(rng) if rng.random() < self.p else 0.0


_HyperparameterT = TypeVar('_HyperparameterT', int, float)


class HyperparameterBounds(TypedDict, Generic[_HyperparameterT]):
    """The lower and upper bounds for sampling a hyperparameter."""

    min: _HyperparameterT
    max: _HyperparameterT


class OptionalHyperparameterBounds(HyperparameterBounds[float]):
    """Hyperparameter sampling bounds for an optimal hyperparameter."""

    optional: bool


class PerModelHyperparameters(TypedDict, total=False):
    """Hyperparameters for one model, overriding the shared hyperparameters."""

    n_blocks: int
    d_block: int
    dropout: float
    numerical_embeddings_d: int
    numerical_embeddings_init_scale: float
    adamw_lr: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_eps: float
    muon_lr: float
    muon_momentum: float
    weight_decay: float
    max_epochs: int | None


def validate_hyperparameter(name: str, value: float | Distribution) -> None:
    if not isinstance(value, Distribution):
        if not math.isfinite(value):
            raise ValueError(f'`{name}` must be finite')

        if name in ('numerical_embeddings_init_scale', 'adamw_lr', 'muon_lr'):
            if value <= 0.0:
                raise ValueError(f'`{name}` must be positive')
        elif name == 'dropout' and value >= 1.0:
            raise ValueError('`dropout` must be less than 1')

        return

    min_value, max_value = value._get_bounds()
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise ValueError(f'The distribution bounds for `{name}` must be finite')

    if name in ('dropout', 'adamw_beta1', 'adamw_beta2', 'muon_momentum'):
        if min_value < 0.0 or min_value > max_value or max_value >= 1.0:
            raise ValueError(
                f'The distribution bounds for `{name}` must be ordered and lie in [0,'
                ' 1)'
            )
    elif name == 'weight_decay':
        if min_value < 0.0:
            raise ValueError(
                'The lower distribution bound for `weight_decay` must be non-negative'
            )
    else:
        if min_value <= 0:
            raise ValueError(
                f'The lower distribution bound for `{name}` must be positive'
            )


# >>> Evaluation


def _accuracy_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
    return metrics.accuracy_pack(y_true=y_true, y_pred=y_pred.argmax(dim=-1))


def _log_loss_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
    return metrics.multiclass_cross_entropy_pack(y_true=y_true, y_proba=y_pred)


def _roc_auc_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
    return metrics.binary_roc_auc_pack(y_true=y_true, y_score=y_pred[..., 1])


def _resolve_metric(
    metric: str, d_out: int
) -> tuple[tabpackage.MetricFn, tabpackage.MetricDirection]:
    allowed_metrics = (
        ('accuracy', 'log_loss', 'roc_auc') if d_out > 1 else ('mse', 'mae', 'r2')
    )
    if metric not in allowed_metrics:
        raise ValueError(f'Unsupported metric for this task: {metric!r}')
    if metric == 'roc_auc' and d_out != 2:
        raise ValueError('ROC AUC requires binary classification')

    metric_fn = {
        'accuracy': _accuracy_pack,
        'log_loss': _log_loss_pack,
        'roc_auc': _roc_auc_pack,
        'mse': metrics.mse_pack,
        'mae': metrics.mae_pack,
        'r2': metrics.r2_pack,
    }[metric]
    metric_direction = (
        tabpackage.MetricDirection.MAXIMIZE
        if metric in ('accuracy', 'roc_auc', 'r2')
        else tabpackage.MetricDirection.MINIMIZE
    )
    return metric_fn, metric_direction


def compute_scalar_metric(
    metric_fn: tabpackage.MetricFn, *, y_true: Tensor, y_pred: Tensor
) -> Tensor:
    return metric_fn(y_true=y_true, y_pred=y_pred.unsqueeze(0))[0]


def _validate_metric_data(
    metric: str, dataset: DatasetName, labels: dict[DatasetName, np.ndarray]
) -> None:
    if dataset not in labels:
        raise ValueError(f'Unknown dataset: {dataset!r}')

    if metric == 'roc_auc' and len(np.unique(labels[dataset])) != 2:
        raise ValueError(f'ROC AUC requires both classes in dataset {dataset!r}')
    elif metric == 'r2' and len(labels[dataset]) < 2:
        raise ValueError(f'R2 requires at least two rows in dataset {dataset!r}')


def resolve_criterion(
    dataset: DatasetName | None,
    metric: str | None,
    *,
    default_dataset: DatasetName | None,
    default_metric: str | None,
    d_out: int,
) -> tabpackage.EvalCriterionConfig | None:
    metric = default_metric if metric is None else metric
    if metric is None or metric == 'none':
        return None

    dataset = default_dataset if dataset is None else dataset
    if dataset is None:
        raise ValueError('An enabled criterion requires an explicit dataset')

    _, metric_direction = _resolve_metric(metric, d_out)
    return tabpackage.EvalCriterionConfig(
        dataset=dataset, metric=metric, metric_direction=metric_direction
    )


def resolve_eval_metrics(
    dataset_names: list[DatasetName] | None,
    metric_names: list[MetricName] | None,
    *,
    eval_criterion: tabpackage.EvalCriterionConfig | None,
    d_out: int,
    labels: dict[DatasetName, np.ndarray],
) -> dict[DatasetName, dict[str, tabpackage.MetricFn]]:
    if eval_criterion is None:
        if (dataset_names is None) != (metric_names is None):
            raise ValueError(
                'Reporting without a criterion requires both datasets and metrics'
            )

        dataset_names = [] if dataset_names is None else dataset_names
        resolved_metric_names = [] if metric_names is None else metric_names
    else:
        dataset_names = (
            [eval_criterion.dataset] if dataset_names is None else dataset_names
        )
        resolved_metric_names = (
            [eval_criterion.metric] if metric_names is None else metric_names
        )

    for dataset in dataset_names:
        if dataset not in labels:
            raise ValueError(f'Unknown dataset: {dataset!r}')

    metric_fns = {
        name: _resolve_metric(name, d_out)[0] for name in resolved_metric_names
    }
    eval_metrics = {
        dataset: metric_fns.copy() for dataset in dataset_names if metric_fns
    }
    if eval_criterion is not None:
        eval_metrics.setdefault(eval_criterion.dataset, {})[eval_criterion.metric] = (
            _resolve_metric(eval_criterion.metric, d_out)[0]
        )

    for dataset, dataset_metrics in eval_metrics.items():
        for metric in dataset_metrics:
            _validate_metric_data(metric, dataset, labels)

    return eval_metrics


# >>> Devices and precision


def resolve_device(device: str | torch.device | Literal['auto']) -> torch.device:
    if device == 'auto':
        device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )

    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')

        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError('The CUDA device index is out of range')

        return torch.device('cuda', index)
    if device.type == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is not available')
        if device.index not in (None, 0):
            raise ValueError('The MPS device index must be unspecified or zero')

        return torch.device('mps')
    if device.type != 'cpu':
        raise ValueError('The device type is not supported')

    return torch.device('cpu')


def resolve_autocast_dtype(dtype: Literal['bfloat16'] | None) -> torch.dtype | None:
    return {None: None, 'bfloat16': torch.bfloat16}[dtype]


# >>> Ensembles


class _Ensemble(BaseEstimator):
    online: bool
    early_stopping_patience: int | None
    algorithm_metric: MetricName | None
    algorithm_dataset: DatasetName | None
    algorithm_device: str | torch.device | Literal['auto']

    def _make_algorithm_config(self) -> KWArgs:
        raise NotImplementedError

    def _make_config(
        self,
        *,
        d_out: int,
        eval_criterion: tabpackage.EvalCriterionConfig | None,
        default_algorithm_dataset: DatasetName | None,
        labels: dict[DatasetName, np.ndarray],
        device: torch.device,
    ) -> tabpackage.EnsembleConfig:
        if self.algorithm_metric is None and eval_criterion is None:
            raise ValueError(
                'Without the evaluation criterion, `algorithm_metric` must be specified'
            )
        if (
            self.algorithm_dataset is None
            and eval_criterion is None
            and default_algorithm_dataset is None
        ):
            raise ValueError(
                'Without the evaluation criterion and with extra datasets passed as'
                ' dictionaries, `algorithm_dataset` must be specified'
            )

        if self.algorithm_metric is None:
            assert eval_criterion is not None
            metric = eval_criterion.metric
        else:
            metric = self.algorithm_metric

        dataset = (
            self.algorithm_dataset
            if self.algorithm_dataset is not None
            else (
                eval_criterion.dataset
                if eval_criterion is not None
                else default_algorithm_dataset
            )
        )
        assert dataset is not None

        _validate_metric_data(metric, dataset, labels)
        metric_fn, metric_direction = _resolve_metric(metric, d_out)

        return tabpackage.EnsembleConfig(
            algorithm=self._make_algorithm_config(),
            online=self.online,
            prediction_transform=(
                partial(torch.clamp_max, max=1.0) if d_out > 1 else None
            ),
            early_stopping_patience=(
                None if eval_criterion is None else self.early_stopping_patience
            ),
            algorithm_metric=metric_fn,
            algorithm_metric_direction=metric_direction,
            algorithm_dataset=dataset,
            algorithm_device=(
                device
                if self.algorithm_device == 'auto'
                else resolve_device(self.algorithm_device)
            ),
        )


class GreedyEnsemble(_Ensemble):
    """Configuration of a greedy ensemble."""

    def __init__(
        self,
        *,
        init_top_k: int = 1,
        max_ensemble_size: int | None = 32,
        with_replacement: bool = False,
        online: bool = True,
        early_stopping_patience: int | None = 32,
        algorithm_metric: MetricName | None = None,
        algorithm_dataset: DatasetName | None = None,
        algorithm_device: str | torch.device | Literal['auto'] = 'auto',
    ) -> None:
        """Greedy ensemble configuration.

        Args:
            init_top_k: the number of top-performing models to start with.
            max_ensemble_size: the maximum number of ensemble members.
            with_replacement: whether a model can be selected more than once.
            online: whether to consider models that are still training.
            early_stopping_patience: the number of updates without improvement
                tolerated before finalizing the ensemble.
            algorithm_metric: the metric used to select ensemble members.
            algorithm_dataset: the dataset used to select ensemble members.
            algorithm_device: the device used to build the ensemble.
        """
        self.init_top_k = init_top_k
        self.max_ensemble_size = max_ensemble_size
        self.with_replacement = with_replacement
        self.online = online
        self.early_stopping_patience = early_stopping_patience
        self.algorithm_metric = algorithm_metric
        self.algorithm_dataset = algorithm_dataset
        self.algorithm_device = algorithm_device

    def _make_algorithm_config(self) -> KWArgs:
        return {
            'type': ensemble.greedy_ensemble,
            'init_top_k': self.init_top_k,
            'max_ensemble_size': self.max_ensemble_size,
            'with_replacement': self.with_replacement,
        }


class TopKEnsemble(_Ensemble):
    """Configuration of an equally weighted ensemble of the top-k performers."""

    def __init__(
        self,
        *,
        k: int,
        online: bool = True,
        early_stopping_patience: int | None = 32,
        algorithm_metric: MetricName | None = None,
        algorithm_dataset: DatasetName | None = None,
        algorithm_device: str | torch.device | Literal['auto'] = 'auto',
    ) -> None:
        """Top-K ensemble configuration.

        Args:
            k: the number of models to select.
            online: whether to consider models that are still training.
            early_stopping_patience: the number of updates without improvement
                tolerated before finalizing the ensemble.
            algorithm_metric: the metric used to rank models.
            algorithm_dataset: the dataset used to rank models.
            algorithm_device: the device used to build the ensemble.
        """
        self.k = k
        self.online = online
        self.early_stopping_patience = early_stopping_patience
        self.algorithm_metric = algorithm_metric
        self.algorithm_dataset = algorithm_dataset
        self.algorithm_device = algorithm_device

    def _make_algorithm_config(self) -> KWArgs:
        return {'type': ensemble.topk_ensemble, 'k': self.k}


class Ensembles(_BaseComposition):
    """A collection of named ensemble configurations."""

    def __init__(
        self, members: dict[str, GreedyEnsemble | TopKEnsemble | KWArgs]
    ) -> None:
        """Named ensemble configurations.

        Each ensemble configuration is accessible through `get_params` and `set_params`
        through names such as `my_ensemble__early_stopping_patience`.
        """
        self.members = members

    @property
    def _ensemble__items(
        self,
    ) -> list[tuple[str, GreedyEnsemble | TopKEnsemble | KWArgs]]:
        return list(self.members.items())

    @_ensemble__items.setter
    def _ensemble__items(
        self, items: list[tuple[str, GreedyEnsemble | TopKEnsemble | KWArgs]]
    ) -> None:
        self.members = dict(items)

    def get_params(self, deep: bool = True) -> KWArgs:
        return self._get_params('_ensemble__items', deep=deep)

    def set_params(self, **params) -> Self:
        if 'members' in params:
            self.members = params.pop('members')
        return self._set_params('_ensemble__items', **params)

    def make_config(
        self,
        *,
        d_out: int,
        eval_criterion: tabpackage.EvalCriterionConfig | None,
        default_algorithm_dataset: DatasetName | None,
        labels: dict[DatasetName, np.ndarray],
        device: torch.device,
    ) -> dict[str, tabpackage.EnsembleConfig]:
        self._validate_names(list(self.members))

        configs = {}
        for name, member in self.members.items():
            if isinstance(member, dict):
                kwargs = member.copy()
                ensemble_type = {'greedy': GreedyEnsemble, 'topk': TopKEnsemble}[
                    kwargs.pop('type', 'greedy')
                ]
                member = ensemble_type(**kwargs)

            configs[name] = member._make_config(
                d_out=d_out,
                eval_criterion=eval_criterion,
                default_algorithm_dataset=default_algorithm_dataset,
                labels=labels,
                device=device,
            )
        return configs
