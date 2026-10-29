"""Optimizers, including the "optimizer pack" pattern.

An optimizer pack is applied to the members of a module pack from `tabpackage.nn` in
parallel, with independent optimizer states and potentially different hyperparameters.
The base class for optimizer packs is `OptimizerPack`.

Note:
    Vanilla non-pack optimizers can be used to optimize module packs when all pack
    members have the same optimization hyperparameters. This is why in many places here
    and in other files, the type annotation for optimizers is `Optimizer` rather than
    `OptimizerPack`, and utilities like those for removing or resetting pack members are
    free functions rather than methods of `OptimizerPack`.

    Using vanilla optimizers in the case of shared hyperparameters can accelerate
    training due to their efficient implementation in PyTorch, particularly when
    `foreach` or the currently experimental `fused` implementation is available, as with
    AdamW. See the official PyTorch documentation for details.
"""

import functools
import math
import typing
import warnings
from collections.abc import Callable
from typing import Any, Final, TypeAlias, TypeVar

import torch
import torch.nn as nn  # noqa: PLR0402
from torch import Tensor
from torch.optim.optimizer import ParamsT

import tabpackage.nn
from tabpackage.nn import PACK_DIM, ParameterPack

_TORCH_IS_OLDER_THAN_2_13 = (
    torch.torch_version.TorchVersion(torch.__version__) < '2.13.0'
)


ScalarGroupValue: TypeAlias = bool | int | float | str | bytes
NumericScalarGroupValue: TypeAlias = bool | int | float
NumericGroupValueList: TypeAlias = list[bool] | list[int] | list[float]

_T = TypeVar('_T')
_ScalarGroupValueT = TypeVar('_ScalarGroupValueT', bound=ScalarGroupValue)
_NumericScalarGroupValueT = TypeVar(
    '_NumericScalarGroupValueT', bound=NumericScalarGroupValue
)


def _check_lr(value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError('The learning rate must be finite and non-negative')


def _check_momentum(value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(
            'Momentum and moving-average decay coefficients must be finite and in the'
            ' range [0, 1)'
        )


def _check_weight_decay(value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError('The weight decay must be finite and non-negative')


def _check_eps(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError('The epsilon value must be finite and positive')


@typing.overload
def _make_group_value(
    value: _ScalarGroupValueT, pack_size: int, device: torch.device | None
) -> _ScalarGroupValueT: ...


@typing.overload
def _make_group_value(
    value: list[_NumericScalarGroupValueT], pack_size: int, device: torch.device | None
) -> Tensor: ...


@typing.overload
def _make_group_value(
    value: Tensor, pack_size: int, device: torch.device | None
) -> Tensor: ...


def _make_group_value(value, pack_size: int, device: torch.device | None):
    if value is None or isinstance(value, ScalarGroupValue):
        return value
    elif isinstance(value, list):
        if len(value) != pack_size:
            raise ValueError(
                'The size of the hyperparameter list must match the pack size'
            )

        return torch.tensor(value, device=device)
    elif isinstance(value, Tensor):
        if value.ndim == 0:
            raise ValueError(
                'Tensors can be used only to specify per-member hyperparameters; use a'
                ' Python scalar to share one value across all members'
            )
        if len(value) != pack_size:
            raise ValueError(
                'The first dimension of a hyperparameter tensor must match the pack'
                ' size'
            )

        # Copy tensors so groups do not share mutable hyperparameter storage.
        return value.to(device, copy=True)
    else:
        raise TypeError(f'Unsupported value type: {type(value)}')


def _get_device(optimizer: torch.optim.Optimizer) -> torch.device | None:
    for group in optimizer.param_groups:
        for p in group['params']:
            return p.device
    return None


def _get_pack_size(optimizer: torch.optim.Optimizer) -> int | None:
    for group in optimizer.param_groups:
        for p in group['params']:
            if isinstance(p, ParameterPack):
                return p.shape[PACK_DIM]
    return None


def _make_defaults(
    keys_values_checks: list[
        tuple[str, ScalarGroupValue | NumericGroupValueList, Callable[..., None]]
    ],
    *,
    pack_size: int,
) -> dict[str, Any]:
    defaults = {}
    for key, value, check_fn in keys_values_checks:
        if isinstance(value, list):
            if len(value) != pack_size:
                raise ValueError(
                    'The size of each hyperparameter list must match the pack size'
                )

            for x in value:
                check_fn(x)
        else:
            check_fn(value)
        defaults[key] = value
    return defaults


class OptimizerPack(torch.optim.Optimizer):
    """The base class for optimizer packs."""

    def __init__(
        self, params: ParamsT, defaults: dict[str, Any], *, pack_size: int
    ) -> None:
        super().__init__(params, defaults)
        for key, value in list(self.defaults.items()):
            self.defaults[key] = _make_group_value(value, pack_size, _get_device(self))

    def add_param_group(self, param_group):
        output = super().add_param_group(param_group)
        group = self.param_groups[-1]

        group_params = group['params']
        if not group_params:
            # Otherwise, it will be impossible to infer the device and pack size below.
            raise ValueError('Empty parameter groups are not supported')

        device = None
        pack_size = None
        for p in group['params']:
            if not isinstance(p, ParameterPack):
                raise TypeError('For now, only parameter packs are supported')

            if device is None:
                device = p.device
                pack_size = p.shape[PACK_DIM]
        assert device is not None
        assert pack_size is not None

        for key, value in list(group.items()):
            if key != 'params':
                group[key] = _make_group_value(value, pack_size, device)

        return output

    def _replace_impl(
        self,
        *,
        pack_idx: Tensor,
        keys_values_checks: list[
            tuple[str, NumericGroupValueList | None, Callable[..., None]]
        ],
    ) -> None:
        tabpackage.nn.check_pack_idx(pack_idx, _get_pack_size(self))

        for _, value, check_fn in keys_values_checks:
            if value is not None:
                if len(value) != len(pack_idx):
                    raise ValueError(
                        'Each replacement hyperparameter list must contain one value'
                        ' per selected pack member'
                    )

                for x in value:
                    check_fn(x)

        for key, value, _ in keys_values_checks:
            if value is None:
                continue

            value_tensor = None
            replaced = False
            for group in self.param_groups:
                current_value = group[key]
                if isinstance(current_value, Tensor):
                    if value_tensor is None:
                        value_tensor = torch.tensor(
                            value,
                            dtype=current_value.dtype,
                            device=current_value.device,
                        )
                    current_value[pack_idx] = value_tensor
                    replaced = True

            if not replaced:
                raise RuntimeError(f'Replacement is not possible for "{key}"')

        optimizer_pack_reset_state(self, pack_idx=pack_idx)


def _make_weight_decay_multiplier(
    *, lr: float | Tensor, weight_decay: float | Tensor
) -> float | Tensor | None:
    return (
        None
        if (
            (isinstance(lr, float) and lr == 0.0)
            or (isinstance(weight_decay, float) and weight_decay == 0.0)
        )
        else (1 - lr * weight_decay)
    )


@typing.overload
def _maybe_unsqueeze(value: Tensor, *, p: Tensor) -> Tensor: ...


@typing.overload
def _maybe_unsqueeze(value: _T, *, p: Tensor) -> _T: ...


def _maybe_unsqueeze(value, *, p):
    return (
        value[(slice(None), *((None,) * (p.ndim - 1)))]
        if isinstance(value, Tensor)
        else value
    )


class AdamWPack(OptimizerPack):
    """A pack of AdamW optimizers."""

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float | list[float],
        # Unlike in `torch.optim.AdamW`, `beta1` and `beta2` are separate arguments,
        # making it easy to specify different values for each pack member by passing
        # lists, as with all other scalar hyperparameters.
        beta1: float | list[float] = 0.9,
        beta2: float | list[float] = 0.999,
        eps: float | list[float] = 1e-8,
        weight_decay: float | list[float],
        pack_size: int,
        support_replacement: bool = False,
    ):
        """
        Args:
            support_replacement: if `True`, the optimizer supports selective replacement
                of current members with new ones. In this case, the underlying step
                counter is not an integer scalar anymore, but a tensor with per-member
                steps, which makes the optimization step a bit slower, and changes the
                results numerically (even if replacements never occur).
        """
        if pack_size <= 0:
            raise ValueError('The pack size must be positive')

        super().__init__(
            params,
            _make_defaults(
                [
                    ('lr', lr, _check_lr),
                    ('beta1', beta1, _check_momentum),
                    ('beta2', beta2, _check_momentum),
                    ('eps', eps, _check_eps),
                    ('weight_decay', weight_decay, _check_weight_decay),
                ],
                pack_size=pack_size,
            ),
            pack_size=pack_size,
        )

        self._support_replacement: Final = support_replacement

    def replace(
        self,
        *,
        pack_idx: Tensor,
        lr: list[float] | None = None,
        beta1: list[float] | None = None,
        beta2: list[float] | None = None,
        eps: list[float] | None = None,
        weight_decay: list[float] | None = None,
    ) -> None:
        """Replace selected members with new ones.

        The method resets the state of the selected optimizers and optionally changes
        their hyperparameters.
        """
        if not self._support_replacement:
            raise RuntimeError(
                'Replacing optimizer pack members requires constructing the optimizer'
                ' with `support_replacement=True`'
            )

        self._replace_impl(
            pack_idx=pack_idx,
            keys_values_checks=[
                ('lr', lr, _check_lr),
                ('beta1', beta1, _check_momentum),
                ('beta2', beta2, _check_momentum),
                ('eps', eps, _check_eps),
                ('weight_decay', weight_decay, _check_weight_decay),
            ],
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # ty: ignore[invalid-method-override]
        """Perform a single optimization step for all pack members."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']

            weight_decay_multiplier = _make_weight_decay_multiplier(
                lr=lr, weight_decay=group['weight_decay']
            )

            for p in group['params']:
                assert isinstance(p, ParameterPack), (
                    'For now, only parameter packs are supported'
                )
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported')

                maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

                state = self.state[p]
                if len(state) == 0:
                    # Initialize the state.
                    state['step'] = (
                        torch.zeros(
                            p.shape[PACK_DIM],
                            dtype=torch.int64,
                            device=p.device,
                        )
                        if self._support_replacement
                        else 0
                    )
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                if weight_decay_multiplier is not None:
                    p.mul_(maybe_unsqueeze(weight_decay_multiplier))

                # Update biased first moment estimate.
                exp_avg.lerp_(grad, maybe_unsqueeze(1 - beta1))

                # Update biased second raw moment estimate.
                if isinstance(beta2, float):
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                else:
                    exp_avg_sq.lerp_(grad.square(), maybe_unsqueeze(1 - beta2))

                # Perform the bias correction.
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr / bias_correction1

                if isinstance(bias_correction2, float):
                    denom = exp_avg_sq.sqrt() / bias_correction2**0.5
                else:
                    denom = exp_avg_sq.sqrt().div_(
                        maybe_unsqueeze(bias_correction2**0.5)
                    )
                denom.add_(maybe_unsqueeze(eps))

                if isinstance(step_size, float):
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p.add_((exp_avg / denom).mul_(maybe_unsqueeze(step_size)), alpha=-1)

        return loss


def compute_muon_update_scale_pack(module: tabpackage.nn.LinearPack) -> Tensor | None:
    """Compute Muon update scaling coefficients for a pack of linear layers."""
    if module.in_features is None and module.out_features is None:
        return None

    if module.in_features is None:
        assert module.out_features is not None
        pack_size = module.out_features.shape[PACK_DIM]
        device = module.out_features.device
    else:
        pack_size = module.in_features.shape[PACK_DIM]
        device = module.in_features.device

    out_f = (
        torch.full((pack_size,), float(module.max_out_features), device=device)
        if module.out_features is None
        else module.out_features.float()
    )
    in_f = (
        torch.full((pack_size,), float(module.max_in_features), device=device)
        if module.in_features is None
        else module.in_features.float()
    )
    return (out_f / in_f).clamp_(min=1.0).sqrt_()


# The following function is a modified version of this function:
# https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py#L5
#
# The modification consists in using `torch.baddbmm` for better efficiency. This
# requires `G` to be three-dimensional.
#
# fmt: off
# ruff: disable[E501]
def _zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim == 3 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c) # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = torch.baddbmm(X, B, X, beta=a)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
# ruff: enable[E501]
# fmt: on


class MuonAdamWPack(OptimizerPack):
    """A pack of optimizers combining Muon and AdamW.

    Muon is intended for weights (excluding biases) of backbones' hidden linear layers,
    and AdamW for everything else. Parameter groups optimized with Muon must be created
    with `make_muon_parameter_group` (see the note below for details).

    Note:
        For parameter packs optimized with Muon, the spectral correction coefficients
        must be provided explicitly, because they depend on the actual per-model
        parameter shapes, which cannot be inferred from parameter packs alone. As a
        consequence, parameter packs with different per-model shapes cannot be
        optimized with Muon within one parameter group, because they will have different
        coefficients. To make the optimizer easier to use correctly, all Muon parameter
        groups are currently required to have no more than one parameter.

    """

    @staticmethod
    def make_muon_parameter_group(module: tabpackage.nn.LinearPack) -> dict[str, Any]:
        """Create a single-parameter group to be optimized with Muon."""
        if (
            module.max_in_features > module.max_out_features
            and module.max_out_features < 16
        ):
            warnings.warn(
                f'The pack of linear layers has shape {module.weight.shape}, which'
                ' looks more like an output layer (should not be optimized with Muon)'
                ' than a backbone layer (can be optimized with Muon). Is this'
                ' intentional?',
                stacklevel=2,
            )

        return {
            'params': [module.weight],
            'muon': True,
            'muon_update_scale': compute_muon_update_scale_pack(module),
        }

    def __init__(
        self,
        params: ParamsT,
        *,
        adamw_lr: float | list[float],
        adamw_beta1: float | list[float] = 0.9,
        adamw_beta2: float | list[float] = 0.999,
        adamw_eps: float | list[float] = 1e-8,
        weight_decay: float | list[float],
        muon_lr: float | list[float],
        muon_momentum: float | list[float] = 0.95,
        muon_newton_schulz_steps: int = 5,
        muon_nesterov: bool = True,
        muon_skip_zero_updates: bool = False,
        pack_size: int,
        support_replacement: bool = False,
        # In practice, the following argument should usually be `True` for better
        # efficiency unless the original parameter gradients must be preserved.
        allow_inplace_grad_mutation: bool,
        compile_muon_newton_schulz: bool = False,
    ):
        """
        Args:
            muon_skip_zero_updates: if `True`, the members with all-zero updates are
                excluded from the Newton-Schulz iterations. On accelerators, this costs
                a device-to-host synchronization per step for each Muon parameter group
                with a gradient. This can speed up training of large packs of
                depth-varying MLPs.
            support_replacement: same as in `AdamWPack`.
        """
        if pack_size <= 0:
            raise ValueError('The pack size must be positive')

        if (
            compile_muon_newton_schulz
            and torch.torch_version.TorchVersion(torch.__version__) < '2.12.0'
        ):
            raise RuntimeError(
                'Compiling the Newton-Schulz iteration for Muon requires PyTorch 2.12.0'
                ' or newer'
            )

        super().__init__(
            params,
            {
                'muon': False,
                'muon_newton_schulz_steps': muon_newton_schulz_steps,
                'muon_nesterov': muon_nesterov,
                'muon_skip_zero_updates': muon_skip_zero_updates,
                **_make_defaults(
                    [
                        ('adamw_lr', adamw_lr, _check_lr),
                        ('adamw_beta1', adamw_beta1, _check_momentum),
                        ('adamw_beta2', adamw_beta2, _check_momentum),
                        ('adamw_eps', adamw_eps, _check_eps),
                        ('weight_decay', weight_decay, _check_weight_decay),
                        ('muon_lr', muon_lr, _check_lr),
                        ('muon_momentum', muon_momentum, _check_momentum),
                    ],
                    pack_size=pack_size,
                ),
            },
            pack_size=pack_size,
        )

        self._support_replacement: Final = support_replacement
        self._allow_inplace_grad_mutation: Final = allow_inplace_grad_mutation
        self._zeropower_via_newtonschulz5 = (
            torch.compile(
                _zeropower_via_newtonschulz5,
                fullgraph=True,
                # Emulate eager precision casts to reduce numerical differences from
                # the non-compiled variant.
                options={'emulate_precision_casts': True},
            )
            if compile_muon_newton_schulz
            else _zeropower_via_newtonschulz5
        )

    def add_param_group(self, param_group):
        output = super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group['muon']:
            if 'muon_update_scale' not in group:
                raise ValueError(
                    'The just added parameter group is configured to be optimized with'
                    ' Muon. In this case, it must have the "muon_update_scale" key'
                    ' with the value computed with the'
                    ' `compute_muon_update_scale_pack` function.'
                )
            if len(group['params']) > 1:
                raise ValueError(
                    'The just added parameter group configured to be optimized with'
                    ' Muon has more than one parameter, which is not allowed'
                )
        return output

    def replace(
        self,
        *,
        pack_idx: Tensor,
        adamw_lr: list[float] | None = None,
        adamw_beta1: list[float] | None = None,
        adamw_beta2: list[float] | None = None,
        adamw_eps: list[float] | None = None,
        weight_decay: list[float] | None = None,
        muon_lr: list[float] | None = None,
        muon_momentum: list[float] | None = None,
        module: nn.Module,
    ) -> None:
        """Replace selected members with new ones.

        See `AdamWPack.replace` for additiona details.

        Important:
            `module` must undergo its member replacement _before_ this method is called,
            because it is used to recompute the new Muon update scales.
        """
        if not self._support_replacement:
            raise RuntimeError(
                'Replacing optimizer pack members requires constructing the optimizer'
                ' with `support_replacement=True`'
            )

        self._replace_impl(
            pack_idx=pack_idx,
            keys_values_checks=[
                ('adamw_lr', adamw_lr, _check_lr),
                ('adamw_beta1', adamw_beta1, _check_momentum),
                ('adamw_beta2', adamw_beta2, _check_momentum),
                ('adamw_eps', adamw_eps, _check_eps),
                ('weight_decay', weight_decay, _check_weight_decay),
                ('muon_lr', muon_lr, _check_lr),
                ('muon_momentum', muon_momentum, _check_momentum),
            ],
        )

        muon_parameter_to_module = {
            x.weight: x
            for x in module.modules()
            if isinstance(x, tabpackage.nn.LinearPack)
        }
        for group in self.param_groups:
            if group['muon']:
                group['muon_update_scale'] = compute_muon_update_scale_pack(
                    muon_parameter_to_module[group['params'][0]]
                )

    def _step_muon(self, group: dict[str, Any]) -> None:
        lr = group['muon_lr']
        momentum = group['muon_momentum']
        weight_decay = group['weight_decay']
        newton_schulz_steps = group['muon_newton_schulz_steps']
        nesterov = group['muon_nesterov']
        skip_zero_updates = group['muon_skip_zero_updates']
        update_scale = group['muon_update_scale']

        weight_decay_multiplier = _make_weight_decay_multiplier(
            lr=lr, weight_decay=weight_decay
        )

        for p in group['params']:
            if p.ndim != 3:  # 1 pack dimension + 2 layer dimensions.
                raise ValueError(
                    'Parameters optimized with Muon must have three dimensions'
                )

            assert isinstance(p, ParameterPack), (
                'For now, only parameter packs are supported'
            )
            if p.grad is None:
                continue

            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Sparse gradients are not supported')

            maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

            state = self.state[p]
            if len(state) == 0:
                # Initialize the state.
                state['muon_momentum_buffer'] = torch.zeros_like(p)
            momentum_buffer: Tensor = state['muon_momentum_buffer']

            if weight_decay_multiplier is not None:
                p.mul_(maybe_unsqueeze(weight_decay_multiplier))

            momentum_buffer.lerp_(grad, maybe_unsqueeze(1 - momentum))
            update = (
                (grad.lerp_ if self._allow_inplace_grad_mutation else grad.lerp)(
                    momentum_buffer, maybe_unsqueeze(momentum)
                )
                if nesterov
                else momentum_buffer
            )
            if skip_zero_updates:
                # All-zero updates do not contribute to the optimization step, and
                # therefore can be safely skipped without participating in the
                # expensive Newton-Schulz iterations. The cost of applying this
                # heuristic is the device-to-host synchronization coming from
                # `torch.nonzero`. It could be avoided by incorporating additional
                # knowledge about the optimized module into the optimizer's state,
                # however, that would make the code significantly more complicated and
                # brittle. The current simple approach already works well.
                assert PACK_DIM == 0
                active_pack_idx = torch.nonzero(update.flatten(1).any(1))[:, 0]
                if len(active_pack_idx) == 0:
                    # All members have all-zero updates, and weight decay and momentum
                    # updates have already been applied.
                    continue
            else:
                active_pack_idx = None
            if active_pack_idx is None or len(active_pack_idx) == len(update):
                update = self._zeropower_via_newtonschulz5(
                    update, steps=newton_schulz_steps
                )
            else:
                active_update = self._zeropower_via_newtonschulz5(
                    update.index_select(0, active_pack_idx), steps=newton_schulz_steps
                )
                update = torch.zeros_like(update, dtype=active_update.dtype)
                if update.device.type == 'mps' and _TORCH_IS_OLDER_THAN_2_13:
                    # The `index_copy_` operation on MPS:
                    # - Fallbacks on CPU prior to PyTorch 2.8.
                    # - Is inefficient prior to PyTorch 2.13.
                    update[active_pack_idx] = active_update
                else:
                    update.index_copy_(0, active_pack_idx, active_update)

            scaled_lr = lr * (
                max(1, grad.size(-2) / grad.size(-1)) ** 0.5
                if update_scale is None
                else update_scale
            )

            assert update.shape == p.shape
            if isinstance(scaled_lr, float):
                p.add_(update, alpha=-scaled_lr)
            else:
                p.addcmul_(update, maybe_unsqueeze(scaled_lr), value=-1)

    def _step_adamw(self, group: dict[str, Any]) -> None:
        lr = group['adamw_lr']
        beta1 = group['adamw_beta1']
        beta2 = group['adamw_beta2']
        eps = group['adamw_eps']
        weight_decay = group['weight_decay']

        weight_decay_multiplier = _make_weight_decay_multiplier(
            lr=lr, weight_decay=weight_decay
        )

        for p in group['params']:
            assert isinstance(p, ParameterPack), (
                'For now, only parameter packs are supported'
            )
            if p.grad is None:
                continue

            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Sparse gradients are not supported')

            maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

            state = self.state[p]
            if len(state) == 0:
                # Initialize the state.
                state['step'] = (
                    torch.zeros(
                        p.shape[PACK_DIM],
                        dtype=torch.int64,
                        device=p.device,
                    )
                    if self._support_replacement
                    else 0
                )
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            state['step'] += 1

            if weight_decay_multiplier is not None:
                p.mul_(maybe_unsqueeze(weight_decay_multiplier))

            # Update biased first moment estimate.
            exp_avg.lerp_(grad, maybe_unsqueeze(1 - beta1))

            # Update biased second raw moment estimate.
            if isinstance(beta2, float):
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            else:
                exp_avg_sq.lerp_(grad.square(), maybe_unsqueeze(1 - beta2))

            # Perform the bias correction.
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            step_size = lr / bias_correction1

            if isinstance(bias_correction2, float):
                denom = exp_avg_sq.sqrt() / bias_correction2**0.5
            else:
                denom = exp_avg_sq.sqrt().div_(maybe_unsqueeze(bias_correction2**0.5))
            denom.add_(maybe_unsqueeze(eps))

            if isinstance(step_size, float):
                p.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                p.add_((exp_avg / denom).mul_(maybe_unsqueeze(step_size)), alpha=-1)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # ty: ignore[invalid-method-override]
        """Perform a single optimization step for all pack members."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['muon']:
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss


# >>> Optimizer utilities


def optimizer_pack_remove(
    optimizer: torch.optim.Optimizer,
    *,
    pack_idx: Tensor,
    old_to_new_parameters: dict[ParameterPack, ParameterPack],
) -> None:
    """Remove some members from an optimizer pack."""
    if not old_to_new_parameters:
        raise ValueError(
            'The mapping from old parameters to new parameters must not be empty'
        )

    pack_size = _get_pack_size(optimizer)
    if pack_size is None:
        raise ValueError('Failed to infer the pack size')

    tabpackage.nn.check_pack_idx(pack_idx, pack_size)

    keep_pack_idx = tabpackage.nn.make_complementary_pack_idx(
        pack_idx, pack_size=pack_size
    )

    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if isinstance(value, Tensor) and value.ndim > 0:
                group[key] = value[keep_pack_idx]
            del key, value

        for i, p in list(enumerate(group['params'])):
            if isinstance(p, ParameterPack):
                state = optimizer.state.pop(p, None)
                # `state` may be missing even if the optimizer has already been used.
                # For example, in `MLPBackbonePack`, some blocks may remain unused
                # (and therefore have no corresponding optimizer state) if the maximum
                # allowed number of blocks is never reached.
                if state is not None:
                    for key, value in list(state.items()):
                        if isinstance(value, Tensor) and value.ndim > 0:
                            state[key] = value[keep_pack_idx]
                        del key, value
                p_new = old_to_new_parameters[p]
                group['params'][i] = p_new
                if state is not None:
                    optimizer.state[p_new] = state


def optimizer_pack_reset_state(
    optimizer: torch.optim.Optimizer, *, pack_idx: Tensor
) -> None:
    """Zero state tensors of specified pack members."""
    tabpackage.nn.check_pack_idx(pack_idx, _get_pack_size(optimizer))

    for group in optimizer.param_groups:
        for p in group['params']:
            if isinstance(p, ParameterPack):
                state = optimizer.state.get(p)
                if state is not None:
                    for value in state.values():
                        if isinstance(value, Tensor) and value.ndim > 0:
                            value[pack_idx] = 0


def should_skip_weight_decay(
    module_name: str, module: nn.Module, parameter_name: str, parameter: nn.Parameter
):
    """Determine if weight decay should be disabled for a given parameter."""
    del module_name
    return (
        # Weight decay is disabled for bias-like parameters, normalization layers, and
        # the first submodules of feature embeddings.
        parameter_name.endswith('bias')
        or isinstance(module, tabpackage.nn.LinearEmbeddingsPack)
        or (
            isinstance(module, tabpackage.nn.CosineEmbeddingsPack)
            and parameter is module.weight
        )
    )
