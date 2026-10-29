"""PyTorch ensembling utilities."""

from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import Tensor

_ScoreFn: TypeAlias = Callable[
    [Tensor],  # Shape: (n_predictions, *)
    Tensor,  # Shape: (n_predictions,)
]


def _validate_predictions(predictions: Tensor) -> None:
    if predictions.ndim < 2:
        raise ValueError(
            'Predictions must have a dimension for ensemble members and at least one'
            " dimension for each member's predictions"
        )
    if predictions.numel() == 0:
        raise ValueError('Predictions must not be empty')


def compute_unsanitized_ensemble_predictions(
    member_predictions: Tensor, member_weights: Tensor | None
) -> Tensor:
    """Aggregate predictions of ensemble members into one _RAW_ prediction.

    Warning:
        Due to the floating-point arithmetic limitations:

        - The computed raw predictions can contain values slightly beyond the range of
          `member_predictions`, i.e. below/above their minimum/maximum values.
          Transform the result as needed, e.g. with `.clamp(max=1.0)` for
          probabilities.
        - If `member_predictions` are probabilities, the computed distributions may
          not sum to `1.0`.

    Args:
        member_predictions: the predictions of the ensemble members.
            Shape: `(n_members, *)`.
        member_weights: weights to aggregate the predictions with, normalized by their
            sum. Pass `None` for equal weights. Shape: `(n_members,)`.
    Returns:
        The aggregated prediction. Shape: `(*)`.
    """
    _validate_predictions(member_predictions)

    if member_weights is None:
        return member_predictions.mean(0)
    else:
        if member_weights.ndim != 1:
            raise ValueError('`member_weights` must be a one-dimensional tensor')
        if len(member_weights) != len(member_predictions):
            raise ValueError(
                'The number of member weights must match the number of ensemble members'
            )
        return (
            (member_weights / member_weights.sum()) @ member_predictions.flatten(1)
        ).view(member_predictions.shape[1:])


def topk_ensemble(predictions: Tensor, *, score_fn: _ScoreFn, k: int) -> Tensor:
    """Build an equally weighted ensemble of the top-k individual performers.

    Args:
        predictions: the predictions to build the ensemble from.
        score_fn: the scoring function taking multiple stacked predictions as input and
            returning their performance scores (one per prediction). Higher scores are
            better. Ties are resolved by selecting the earliest predictions.
        k: the number of predictions to select, capped by the number available.
    Returns:
        The selected prediction indices.
    """
    _validate_predictions(predictions)
    if k <= 0:
        raise ValueError('`k` must be positive to select at least one ensemble member')

    prediction_scores = score_fn(predictions)
    if prediction_scores.ndim != 1:
        raise ValueError(
            'The scoring function must return a one-dimensional tensor of scores'
        )
    if len(prediction_scores) != len(predictions):
        raise ValueError('The scoring function must return one score per prediction')

    return (
        torch.argmax(prediction_scores)[None]
        if k == 1
        else torch.argsort(prediction_scores, descending=True, stable=True)[:k]
    )


def greedy_ensemble(
    predictions: Tensor,
    *,
    score_fn: _ScoreFn,
    init_top_k: int = 1,
    max_ensemble_size: int | None = None,
    with_replacement: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Build an ensemble by greedily adding the best candidate one at a time.

    The algorithm initializes the ensemble by unconditionally selecting the top-k
    predictions based on their individual performance, and then greedily adds more
    predictions one at a time, stopping when reaching the allowed ensemble size or when
    adding more predictions does not improve performance.

    Args:
        predictions: the predictions to build the ensemble from.
        score_fn: the scoring function taking multiple stacked predictions as input and
            returning their performance scores (one per prediction). Higher scores are
            better. Usually, it is based on the main task metric (e.g., accuracy,
            _negative_ RMSE, etc.). The algorithm stops when the ensemble performance
            stops improving according to this score.
        init_top_k: the number of the best individual performers to unconditionally add
            to the ensemble on initialization.
        max_ensemble_size: the maximum allowed ensemble size. When
            `with_replacement=False`, it is the maximum number of distinct ensemble
            members. Otherwise, it is the maximum number of member selections that the
            algorithm performs, including the unconditional selections in the beginning.
            Must be specified when `with_replacement=True`.
        with_replacement: if `True`, each prediction can be added to the ensemble any
            number of times.
    Returns:
        The selected indices in original prediction order, and their selection counts
        when `with_replacement=True`; otherwise, the second item is `None`.
    """
    _validate_predictions(predictions)
    if not torch.is_floating_point(predictions):
        raise TypeError('Predictions must be floating-point values')
    if init_top_k <= 0:
        raise ValueError(
            '`init_top_k` must be positive to initialize the ensemble with at least one'
            ' member'
        )
    if with_replacement and max_ensemble_size is None:
        raise ValueError(
            'Set `max_ensemble_size` to limit the number of ensemble members when'
            ' `with_replacement=True`'
        )
    if max_ensemble_size is not None and init_top_k > max_ensemble_size:
        raise ValueError('`init_top_k` must not exceed `max_ensemble_size`')

    device = predictions.device

    # Individual candidate scores.
    prediction_scores = score_fn(predictions)

    # Initial ensemble members.
    init_idx = topk_ensemble(
        predictions, score_fn=lambda _: prediction_scores, k=init_top_k
    )

    # Prediction-related constants.
    n_predictions = len(predictions)
    prediction_range_idx = torch.arange(n_predictions, device=device)

    # The mask indicating if a given prediction is selected as an ensemble member.
    ensemble_mask = torch.zeros(n_predictions, dtype=torch.bool, device=device)
    ensemble_mask[init_idx] = True

    if with_replacement:
        # Each prediction's weight is the number of times it was selected.
        ensemble_weights = torch.zeros(
            n_predictions, dtype=predictions.dtype, device=device
        )
        ensemble_weights[init_idx] = 1.0
    else:
        ensemble_weights = None

    # Current ensemble prediction.
    ensemble_prediction = predictions[init_idx].mean(0)

    # Current ensemble score.
    ensemble_score = score_fn(ensemble_prediction[None])[0]

    ensemble_size = len(init_idx)  # The number of distinct ensemble members.
    ensemble_weight_sum = ensemble_size

    # Without replacement, the weight sum equals the number of distinct members.
    while ensemble_weight_sum < (
        n_predictions
        if max_ensemble_size is None  # True only when `with_replacement=False`.
        else max_ensemble_size
        if with_replacement
        else min(n_predictions, max_ensemble_size)
    ):
        # Get the indices of predictions available for selection.
        candidate_idx = (
            prediction_range_idx
            if with_replacement
            else torch.nonzero_static(
                ~ensemble_mask, size=n_predictions - ensemble_size
            )[:, 0]
        )

        # Compute the ensemble predictions for all possible ensembles formed by adding
        # one of the candidate predictions.
        candidate_predictions = (
            predictions.clone()
            if with_replacement
            else predictions.index_select(0, candidate_idx)
        )
        candidate_predictions.lerp_(
            ensemble_prediction[None], ensemble_weight_sum / (ensemble_weight_sum + 1)
        )

        # Compute ensemble scores of all possible ensembles.
        candidates_scores = score_fn(candidate_predictions)

        # Get the best possible ensemble score.
        best_candidate_score = candidates_scores.max()
        # If the obtained score is no better than that of the current ensemble, stop the
        # algorithm.
        if best_candidate_score <= ensemble_score:
            break

        # Find all candidates with the best score.
        candidate_local_mask = candidates_scores == best_candidate_score
        best_candidate_local_idx = torch.nonzero(candidate_local_mask)[:, 0]
        best_candidate_idx = candidate_idx[candidate_local_mask]
        if len(best_candidate_idx) == 0:
            raise ValueError(
                'Cannot select the best ensemble candidate because the scoring function'
                ' returned a NaN score'
            )

        if len(best_candidate_idx) == 1:
            # Select the only candidate with the best score.
            best_candidate_local_index = best_candidate_local_idx[0]
            best_candidate_index = best_candidate_idx[0]
        else:
            # Break ties by individual score, then by the original prediction order.
            best_candidate_local_index = best_candidate_local_idx[
                torch.argmax(prediction_scores[best_candidate_idx])
            ]
            best_candidate_index = candidate_idx[best_candidate_local_index]

        # Update the ensemble state.
        ensemble_mask[best_candidate_index] = True
        if ensemble_weights is not None:
            ensemble_weights[best_candidate_index] += 1.0
        ensemble_score = candidates_scores[best_candidate_local_index]
        # Clone the only needed prediction and delete the rest to leave more free memory
        # for the next iteration.
        ensemble_prediction = candidate_predictions[best_candidate_local_index].clone()
        del candidate_predictions
        if (
            ensemble_weights is None
            or ensemble_weights[best_candidate_index].item() == 1.0
        ):
            ensemble_size += 1
        ensemble_weight_sum += 1

        if not with_replacement:
            assert ensemble_size == ensemble_weight_sum

    ensemble_idx = torch.nonzero_static(ensemble_mask, size=ensemble_size)[:, 0]
    return ensemble_idx, (
        None if ensemble_weights is None else ensemble_weights[ensemble_idx]
    )
