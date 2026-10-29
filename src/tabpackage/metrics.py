"""Metric computation using PyTorch.

Common conventions:

- The pack dimension must be `tabpackage.nn.PACK_DIM`.
- The `*_pack` functions (`accuracy_pack`, `mse_pack`, etc.) compute common metrics for
  prediction packs; their distinct property is that the pack dimension is preserved
  during the metric aggregation.
- The argument naming follows that of Scikit-learn 1.9+. In particular, predicted
  labels are accepted as `y_pred`, predicted probabilities as `y_proba` and decision
  scores (the higher - the better) as `y_score`.
- Prediction tensors are expected to have exactly one pack dimension.
- Ground-truth labels with omitted pack and batch dimensions are broadcast along these
  dimensions to match predictions. This in particular means that labels without the pack
  dimension are shared between all pack members.
"""

import torch
import torch.nn as nn  #  noqa: PLR0402
from torch import Tensor

import tabpackage.utils.torch
from tabpackage.nn import PACK_DIM


def _check_y_shapes(*, y_true: Tensor, y_pred: Tensor, n_trailing_dims: int) -> None:
    assert n_trailing_dims >= 0

    if y_pred.ndim < 1 + n_trailing_dims:
        raise ValueError(
            'The prediction tensor has too few dimensions to represent a pack of'
            ' outputs'
        )
    if y_true.ndim > y_pred.ndim - n_trailing_dims:
        raise ValueError('Labels have too many dimensions for the provided predictions')

    if y_true.ndim > 0:
        # Only missing leading dimensions may be broadcast; existing ones must match.
        end = y_pred.ndim - n_trailing_dims
        if y_true.shape != y_pred.shape[end - y_true.ndim : end]:
            raise ValueError(
                'The label and prediction shapes are incompatible; only missing leading'
                ' label dimensions may be broadcast'
            )

    if y_pred.numel() == 0:
        raise ValueError('Predictions must not be empty')


def _mean_nonpack(x: Tensor) -> Tensor:
    assert x.ndim >= 1  # Must have at least the pack dimension.

    if x.ndim == 1:
        # Relying on the `else` branch would be incorrect, because it would result in
        # `x.mean(())`, which means aggregating across all dimensions in PyTorch.
        # The only remaining dimension must be the pack dimension.
        assert PACK_DIM == 0
        return x  # No dimensions to aggregate across.
    else:
        nonpack_dims = tuple(dim for dim in range(x.ndim) if dim != PACK_DIM)
        return x.mean(nonpack_dims)


# The implementation follows `_mean_nonpack`.
def _sum_nonpack(x: Tensor) -> Tensor:
    assert x.ndim >= 1

    if x.ndim == 1:
        assert PACK_DIM == 0
        return x
    else:
        nonpack_dims = tuple(dim for dim in range(x.ndim) if dim != PACK_DIM)
        return x.sum(nonpack_dims)


def accuracy_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the accuracy for each pack member."""
    if not tabpackage.utils.torch.is_integer(y_true):
        raise TypeError('Labels must have an integer data type')
    if y_true.dtype != y_pred.dtype:
        raise TypeError('Labels and predictions must have the same data type')
    _check_y_shapes(y_true=y_true, y_pred=y_pred, n_trailing_dims=0)

    return _mean_nonpack((y_true == y_pred).float())


def _midranks_pack(x: Tensor) -> Tensor:
    """Compute the one-based ranks of sorted values, averaged within tied groups.

    For example, the midranks of `[10, 20, 20, 20, 30]` are `[1, 3, 3, 3, 5]`: the
    three tied values occupy the ranks 2, 3 and 4, hence the average rank 3.

    The input must be sorted along the last (batch) dimension in the ascending order.
    """
    assert x.ndim == 2

    n_objects = x.shape[-1]
    indices = torch.arange(n_objects, device=x.device)
    ones = torch.ones((x.shape[PACK_DIM], 1), dtype=torch.bool, device=x.device)

    changes = x[:, 1:] != x[:, :-1]
    # Whether a position starts a tied group.
    starts = torch.cat([ones, changes], dim=-1)
    # Whether a position ends a tied group.
    ends = torch.cat([changes, ones], dim=-1)

    # For each position, the first and the last position of its tied group. The running
    # extrema propagate the group boundaries to all the positions within the group.
    first = torch.cummax(torch.where(starts, indices, -1), dim=-1).values
    last = torch.cummin(
        torch.where(ends, indices, n_objects).flip(-1), dim=-1
    ).values.flip(-1)
    # `+ 1` turns the zero-based positions into one-based ranks.
    return (first + last).to(x.dtype) / 2 + 1


def binary_roc_auc_pack(*, y_true: Tensor, y_score: Tensor) -> Tensor:
    """Compute the ROC AUC for each pack member."""
    # Because of the implementation complexity, the supported input data types and
    # shapes are limited to those that were actually tested.
    if y_true.dtype != torch.int64:
        raise TypeError('Labels must have the data type `torch.int64`')
    if y_score.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            'Scores must have the data type `torch.float32` or `torch.float64`'
        )

    if y_true.ndim != 1:
        raise ValueError('The label tensor must have one dimension')
    if y_score.ndim != 2:
        raise ValueError('The score tensor must have two dimensions')
    if y_true.shape[-1] != y_score.shape[-1]:
        raise ValueError('Labels and scores must have the same batch size')
    if y_score.numel() == 0:
        raise ValueError('Scores must not be empty')

    sorted_indices = torch.argsort(y_score, dim=-1, stable=True)
    y_score_sorted = torch.take_along_dim(y_score, sorted_indices, dim=-1)
    y_true_sorted = y_true[sorted_indices]

    n_positives = y_true.sum()
    n_negatives = len(y_true) - n_positives
    n_pairs = n_positives * n_negatives

    # The U statistic is the number of the (positive, negative) pairs ranked correctly,
    # with the ties counted as one half.
    positive_rank_sum = (_midranks_pack(y_score_sorted) * y_true_sorted).sum(-1)
    u_statistic = positive_rank_sum - n_positives * (n_positives + 1) / 2

    return torch.where(n_pairs == 0, torch.nan, u_statistic / n_pairs).to(y_score.dtype)


def binary_cross_entropy_pack(*, y_true: Tensor, y_proba: Tensor) -> Tensor:
    """Compute the binary cross-entropy for each pack member."""
    if not tabpackage.utils.torch.is_integer(y_true):
        raise TypeError('Labels must have an integer data type')
    if not torch.is_floating_point(y_proba):
        raise TypeError('Probabilities must be floating-point values')
    _check_y_shapes(y_true=y_true, y_pred=y_proba, n_trailing_dims=0)

    return _mean_nonpack(
        nn.functional.binary_cross_entropy(
            input=y_proba,
            target=y_true.to(y_proba.dtype).expand_as(y_proba),
            reduction='none',
        )
    )


def multiclass_cross_entropy_pack(*, y_true: Tensor, y_proba: Tensor) -> Tensor:
    """Compute the multiclass cross-entropy for each pack member."""
    # Only class labels are supported as `y_true`; plus, this tensor is used to index
    # another tensor. Hence the `torch.int64` requirement.
    if y_true.dtype != torch.int64:
        raise TypeError('Labels must have the int64 data type')
    if not torch.is_floating_point(y_proba):
        raise TypeError('Probabilities must be floating-point values')
    _check_y_shapes(y_true=y_true, y_pred=y_proba, n_trailing_dims=1)

    correct_class_probs = y_proba.gather(
        -1, y_true.expand(y_proba.shape[:-1]).unsqueeze(-1)
    ).squeeze(-1)
    return _mean_nonpack(
        torch.log(correct_class_probs)
        .clamp_min(-100.0)  # The clamping follows `torch.nn.BCELoss`.
        .neg()
    )


def mse_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the mean squared error for each pack member."""
    if not torch.is_floating_point(y_true):
        raise TypeError('Labels must be floating-point values')
    if y_true.dtype != y_pred.dtype:
        raise TypeError('Labels and predictions must have the same data type')
    _check_y_shapes(y_true=y_true, y_pred=y_pred, n_trailing_dims=0)

    return _mean_nonpack(
        nn.functional.mse_loss(y_pred, y_true.expand_as(y_pred), reduction='none')
    )


def mae_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the mean absolute error for each pack member."""
    if not torch.is_floating_point(y_true):
        raise TypeError('Labels must be floating-point values')
    if y_true.dtype != y_pred.dtype:
        raise TypeError('Labels and predictions must have the same data type')
    _check_y_shapes(y_true=y_true, y_pred=y_pred, n_trailing_dims=0)

    return _mean_nonpack(
        nn.functional.l1_loss(y_pred, y_true.expand_as(y_pred), reduction='none')
    )


def r2_pack(*, y_true: Tensor, y_pred: Tensor, force_finite: bool = True) -> Tensor:
    """Compute the R2 score for each pack member."""
    if not torch.is_floating_point(y_true):
        raise TypeError('Labels must be floating-point values')
    if y_true.dtype != y_pred.dtype:
        raise TypeError('Labels and predictions must have the same data type')
    _check_y_shapes(y_true=y_true, y_pred=y_pred, n_trailing_dims=0)

    numerator = _sum_nonpack((y_pred - y_true).square())
    if y_true.ndim < y_pred.ndim:
        shape_without_pack_dim = torch.Size(
            [x for i, x in enumerate(y_pred.shape) if i != PACK_DIM]
        )
        denominator = torch.sum(
            torch.broadcast_to(
                (y_true - y_true.mean()).square(), shape_without_pack_dim
            )
        )
    else:
        denominator = _sum_nonpack(
            (
                y_true
                - _mean_nonpack(y_true).reshape(
                    [-1 if dim == PACK_DIM else 1 for dim in range(y_true.ndim)]
                )
            ).square()
        )
    ratio = numerator / denominator

    if force_finite:
        # Map constant-target cases to scores of 1 (perfect) or 0 before checking
        # the sample count below.
        ratio = torch.where(
            numerator == 0.0, 0.0, torch.where(denominator == 0.0, 1.0, ratio)
        )

    n_objects = y_pred.numel() // y_pred.shape[PACK_DIM]
    return torch.full_like(ratio, torch.nan) if n_objects < 2 else 1.0 - ratio
