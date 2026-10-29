"""Neural network modules, including the "Module Pack" pattern.

A module pack is a set of same-family modules with potentially different
hyperparameters that are stored and applied as one module, which makes it possible to
train many models in parallel efficiently. The base class for module packs is
`ModulePack`. The pack dimension of all tensors of a module pack is `PACK_DIM`.
"""

import abc
import contextlib
import enum
import inspect
import itertools
import math
import typing
from collections import OrderedDict
from typing import Any, Final, NamedTuple, TypeAlias, TypeVar

import torch
import torch.nn as nn  #  noqa: PLR0402
import torch.optim.optimizer
from torch import Tensor

_TORCH_IS_OLDER_THAN_2_15 = torch.torch_version.TorchVersion(torch.__version__) < '2.15'

# In many places, the implementation _implicitly_ assumes the following values
# and is _not_ ready for other values. Thus, these constants exist only for additional
# clarity where possible, and must never be changed.
PACK_DIM: Final = 0
BATCH_DIM: Final = 1


KWArgs: TypeAlias = dict[str, Any]


# >>> Generic modules


class PackView(nn.Module):
    """Turn a tensor to a valid input for a module pack.

    This is a helper module for module packs that is not itself a module pack, but
    follows some of the conventions described in `ModulePack`.

    .. note::
        The module uses `torch.Tensor.expand` under the hood and thus does not create
        any copies.

    **Shape**

    - Input: `(batch_size, d)` or `(pack_size, batch_size, d)`
    - Output: `(pack_size, batch_size, d)`

    **Examples**

    Two-dimensional tensors are expanded to three-dimensional tensors holding `k`
    identical views of the original input:

    >>> m = PackView(pack_size=2)
    >>> x = torch.randn(3, 4)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    >>> torch.all(m(x) == x.unsqueeze(PACK_DIM)).item()
    True

    Three-dimensional tensors are accepted only in training mode and returned as-is:

    >>> m = PackView(pack_size=2)
    >>> x = torch.randn(2, 3, 4)
    >>> m(x) is x
    True
    """

    def __init__(self, *, pack_size: int) -> None:
        super().__init__()
        self._pack_size = pack_size

    @property
    def pack_size(self) -> int:
        """The pack size."""
        return self._pack_size

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        pack_size = self.pack_size if pack_idx is None else len(pack_idx)
        if x.ndim == 2:
            x = x.unsqueeze(PACK_DIM).expand(pack_size, -1, -1)
        else:
            if x.ndim != 3:
                raise ValueError('The input must have two or three dimensions')
            if not self.training:
                raise RuntimeError('Three-dimensional input requires training mode')
            if x.shape[PACK_DIM] != pack_size:
                raise ValueError(
                    'The pack dimension of the input must match the selected pack size'
                )
        return x


# >>> Tensor packs


def _check_idx(idx: Tensor) -> None:
    if not (idx.dtype == torch.int32 or idx.dtype == torch.int64):
        raise TypeError(
            'Indices must have the data type `torch.int32` or `torch.int64`'
        )
    if idx.ndim != 1:
        raise ValueError('Indices must be one-dimensional')
    if len(idx) == 0:
        raise ValueError('Indices must not be empty')


def check_pack_idx(pack_idx: Tensor, pack_size: int | None) -> None:
    """Check basic requirements for indexing pack members."""
    # Only cheap checks are performed. In particular, the values of `pack_idx`
    # are not checked, because that would require a device-to-host synchronization.
    _check_idx(pack_idx)
    if pack_size is not None and len(pack_idx) > pack_size:
        raise ValueError('The number of pack indices must not exceed the pack size')


def make_complementary_pack_idx(pack_idx: Tensor, *, pack_size: int) -> Tensor:
    """Make the indices of the pack members that are not in `pack_idx`."""
    check_pack_idx(pack_idx, pack_size)

    mask = torch.ones(pack_size, dtype=torch.bool, device=pack_idx.device)
    mask[pack_idx] = False
    # The function could have a keyword argument like this:
    # `assume_unique_nonnegative_pack_idx: bool = False`; so that when it is `True`,
    # then `torch.nonzero_static` is used to avoid the device-to-host synchronization.
    # However, to keep it simple and less error-prone for users, the argument is not
    # added for now.
    return torch.nonzero(mask)[:, 0]


@typing.overload
def _get_tensor_pack(x: None, pack_idx: Tensor | None) -> None: ...


@typing.overload
def _get_tensor_pack(x: Tensor, pack_idx: Tensor | None) -> Tensor: ...


def _get_tensor_pack(x, pack_idx):
    if x is None or pack_idx is None:
        return x

    _check_idx(pack_idx)
    # `index_select` seems to be a little bit faster for the expected workloads than the
    # square-brackets-based indexing. However, it requires nonnegative indices.
    return x.index_select(
        PACK_DIM, torch.where(pack_idx < 0, pack_idx + x.shape[PACK_DIM], pack_idx)
    )


class ParameterPack(nn.Parameter):
    """Parameter pack.

    Module packs _must_ use this class to store their trainable parameters.
    """


class _BufferPackMeta(nn.parameter._BufferMeta):  # ty: ignore[unresolved-attribute]
    """
    `BufferPack` relies on this metaclass for `isinstance(..., BufferPack)` to work
    as one would expect. Otherwise, because of `torch.nn.Buffer` implementation details,
    the following (unexpected) behavior would take place:

    >>> class NaiveBufferPack(nn.Buffer):
    ...     pass
    ...
    >>> naive_buffer_pack = NaiveBufferPack(torch.zeros(1))
    >>> isinstance(naive_buffer_pack, NaiveBufferPack)
    False
    """

    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or (
            self is BufferPack
            and isinstance(instance, torch.Tensor)
            and getattr(instance, '_is_buffer_pack', False)
        )


@typing.final
class BufferPack(nn.Buffer, metaclass=_BufferPackMeta):
    """Buffer pack.

    Module packs _must_ use this class to store their non-trainable tensors.
    """

    def __new__(cls, *args, **kwargs):
        t = nn.Buffer.__new__(cls, *args, **kwargs)
        t._is_buffer_pack = True  # ty: ignore[unresolved-attribute]
        return t

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError('Subclassing is not allowed')


_TensorPackT = TypeVar('_TensorPackT', bound='ParameterPack | BufferPack')


def _make_tensor_pack_like(x: _TensorPackT, *, data: Tensor) -> _TensorPackT:
    # `type` works as expected for `ParameterPack` and inherited classes, but not for
    # `BufferPack`. This is why inheriting from `BufferPack` is not allowed.
    cls = type(x) if isinstance(x, ParameterPack) else BufferPack
    return cls(data).requires_grad_(x.requires_grad)  # ty: ignore[invalid-return-type]


def _load_tensor_packs_from_state_dict_at(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    *,
    pack_idx: Tensor,
    state_dict_idx: Tensor | None,
) -> None:
    with torch.no_grad():
        for name, x in itertools.chain(
            module.named_parameters(recurse=False, remove_duplicate=False),
            (
                (name, value)
                for name, value in module.named_buffers(
                    recurse=False, remove_duplicate=False
                )
                if name not in module._non_persistent_buffers_set
            ),
        ):
            if not isinstance(x, ParameterPack | BufferPack):
                raise TypeError(
                    'The module contains parameters or buffers that do not support pack'
                    ' operations'
                )

            key = prefix + name
            value = state_dict[key]

            if not isinstance(value, Tensor):
                raise TypeError('State dictionary values must be tensors')
            if value.ndim != x.ndim:
                raise ValueError(
                    'The source and destination tensors have incompatible shapes'
                )
            if value.shape[1:] != x.shape[1:]:
                raise ValueError(
                    'The source and destination tensors have incompatible shapes'
                )
            if state_dict_idx is None and value.shape[PACK_DIM] != len(pack_idx):
                raise ValueError(
                    'The state dictionary contains a different number of pack members'
                    ' than the destination selection'
                )

            x[pack_idx] = value if state_dict_idx is None else value[state_dict_idx]


# >>> Module packs


class _ModulePackBuffers(OrderedDict):
    """
    This class ensures that reassigning a module pack's field that is a buffer pack
    will result in a field that is also a buffer pack.
    """

    def __setitem__(self, key, value) -> None:
        current_value = self.get(key)
        if (
            current_value is not None
            # The following checks are the reason why `_BufferPackMeta` is needed.
            and isinstance(current_value, BufferPack)
            and not isinstance(value, BufferPack)
        ):
            value = BufferPack(value)
        return super().__setitem__(key, value)


class ModulePack(nn.Module, abc.ABC):
    """The base class for module packs.

    A module pack is a set of modules of the same family that may differ in their
    hyperparameters.

    **Examples**

    - `DropoutPack`: a pack of dropout layers with different dropout ratios.
    - `LinearPack`: a pack of linear layers with different input and output shapes.

    **Conventions**

    - Module packs _must_ use `ParameterPack` and `BufferPack` classes to store
      parameters and buffers rather than `torch.nn.Parameter` and `torch.nn.Buffer`.
      Otherwise, the behavior will be incorrect.
    - Module pack constructors _must_ take a required `pack_size: int` argument denoting
      the number of modules in the pack.
    - Module pack constructors _must_ take per-pack-member hyperparameters as lists of
      per-pack-member hyperparameter values.
    - Module packs _may_ implement the `replace` method for replacing some of the
      current members with new ones, with potentially different hyperparameters and
      reinitialized parameters. When doing this during training, also call the optimizer
      pack's `replace` method for the same members to reset their optimizer state.
    - Module pack forward passes _may_ take an additional `pack_idx: Tensor | None`
      argument to perform a forward pass with a selected set of pack members. Inputs
      with a pack dimension must already follow the order and repetitions in `pack_idx`;
      the forward pass selects module state, not input rows. Note that selecting pack
      members with `pack_idx` copies their selected entries. Therefore, this mechanism
      should be used only when this overhead is acceptable, or when there are no
      alternatives (e.g., for performing a training step for a subset of pack members).
      As an example of the second use case, `MLPBackbonePack` currently relies on this
      mechanism for depth-varying MLPs. Examples of use cases where `pack_idx` should
      _not_ be used: (1) permanently disabling some of the pack members; instead, use
      `ModulePack.remove`; (2) evaluating some of the pack members; instead, use
      `ModulePack.temporarily_select`.
    """

    def __init__(self, *args, pack_size: int, **kwargs) -> None:
        del pack_size
        super().__init__(*args, **kwargs)
        self._buffers = _ModulePackBuffers()

    @property
    @abc.abstractmethod
    def pack_size(self) -> int:
        """The pack size."""

    def try_get_pack_size(self) -> int | None:
        """Get the pack size, or `None` if it cannot be inferred."""
        try:
            return self.pack_size
        except RuntimeError:
            return None

    def load_state_dict_at(
        self,
        state_dict: dict[str, Tensor],
        *,
        pack_idx: Tensor,
        state_dict_idx: Tensor | None = None,
    ) -> None:
        """Load state dictionary for specific pack members.

        Source rows are matched positionally to destination rows selected by `pack_idx`.
        If `state_dict_idx` is omitted, source tensors must contain exactly
        `len(pack_idx)` members; otherwise, `state_dict_idx` selects the source rows.

        Warning:
            This method does _not_ call the hooks registered for `load_state_dict`.
        """
        _check_module_pack(self)
        check_pack_idx(pack_idx, self.pack_size)
        if state_dict_idx is not None:
            _check_idx(state_dict_idx)
            if pack_idx.shape != state_dict_idx.shape:
                raise ValueError(
                    'The source and destination selections contain different numbers of'
                    ' pack members'
                )
        if self.state_dict().keys() != state_dict.keys():
            raise ValueError(
                'The state dictionary has missing or unexpected keys for this module'
            )

        def load(
            module: nn.Module, local_state_dict: dict[str, Tensor], prefix: str
        ) -> None:
            (
                type(module)._load_from_state_dict_at
                if isinstance(module, ModulePack)
                else _load_tensor_packs_from_state_dict_at
            )(
                module,  # ty: ignore[invalid-argument-type]
                local_state_dict,
                prefix,
                pack_idx=pack_idx,
                state_dict_idx=state_dict_idx,
            )

            # `.named_children()` is not used, because it removes duplicates.
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict, '')

    def _load_from_state_dict_at(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        *,
        pack_idx: Tensor,
        state_dict_idx: Tensor | None,
    ) -> None:
        _load_tensor_packs_from_state_dict_at(
            self,
            state_dict,
            prefix,
            pack_idx=pack_idx,
            state_dict_idx=state_dict_idx,
        )

    def remove(self, *, pack_idx: Tensor) -> dict[ParameterPack, ParameterPack]:
        """Remove some members from the module pack.

        The method replaces the current parameter packs and buffer packs with smaller
        ones. To continue training, pass the returned parameter mapping and the same
        `pack_idx` to `tabpackage.optim.optimizer_pack_remove` to update the optimizer.

        Args:
            pack_idx: the pack member indices to remove.
        Returns:
            The mapping from the replaced parameter packs to the new ones.
        """
        _check_module_pack(self)
        check_pack_idx(pack_idx, self.pack_size)
        # Duplicate indices would make the `PackView` size updates incorrect, so the
        # cost of device-to-host synchronization is paid here to avoid the issue.
        if len(pack_idx) != len(torch.unique(pack_idx)):
            raise ValueError('Pack indices to remove must be unique')

        keep_pack_idx = None
        old_to_new_parameters = {}
        for name, x in itertools.chain(self.named_parameters(), self.named_buffers()):
            if isinstance(x, ParameterPack | BufferPack):
                if keep_pack_idx is None:
                    keep_pack_idx = make_complementary_pack_idx(
                        pack_idx, pack_size=x.shape[PACK_DIM]
                    )

                if '.' in name:
                    submodule_name, attr = name.rsplit('.', 1)
                    submodule = self.get_submodule(submodule_name)
                else:
                    submodule = self
                    attr = name

                new_x = _make_tensor_pack_like(x, data=x.data[keep_pack_idx])
                if isinstance(x, BufferPack):
                    submodule.register_buffer(
                        attr,
                        new_x,
                        persistent=attr not in submodule._non_persistent_buffers_set,
                    )
                else:
                    setattr(submodule, attr, new_x)

                if isinstance(x, ParameterPack):
                    assert isinstance(new_x, ParameterPack)
                    old_to_new_parameters[x] = new_x

        for submodule in self.modules():
            if isinstance(submodule, PackView):
                submodule._pack_size -= len(pack_idx)

        return old_to_new_parameters

    # A potential alternative to the following method could be `ModulePack.index`
    # that would produce a copy of the original module with selected pack members.
    # Contrary to that hypothetical method, this method performs the minimal possible
    # number of copy operations (one per tensor pack).
    @contextlib.contextmanager
    def temporarily_select(self, *, pack_idx: Tensor):
        """Temporarily use specified pack members in evaluation mode.

        Args:
            pack_idx: the indices of the pack members to use.

        The selection is performed literally based on the provided `pack_idx`. That is,
        the order of the new pack members will follow that of `pack_idx`, and repeating
        values in `pack_idx` means repeating pack members, which in particular can
        result in a module pack of a larger size than the original.

        Technically, the method temporarily replaces all parameter packs and buffer
        packs with new ones. This in particular means that all in-place modifications of
        the tensor packs within the context are discarded on exit.
        """
        _check_module_pack(self)
        check_pack_idx(pack_idx, None)

        for submodule in self.modules():
            # Since the method replaces the original tensor objects with new ones,
            # which in particular means disconnecting the module from its optimizer,
            # training the module in its temporary state is rather an unusual use case.
            if submodule.training:
                raise RuntimeError(
                    'The module and all its submodules must be in evaluation mode to'
                    ' use this method'
                )

        original_modules_and_tensors = []
        original_pack_views = []

        try:
            for name, x in itertools.chain(
                self.named_parameters(), self.named_buffers()
            ):
                if isinstance(x, ParameterPack | BufferPack):
                    if '.' in name:
                        submodule_name, attr = name.rsplit('.', 1)
                        submodule = self.get_submodule(submodule_name)
                    else:
                        submodule = self
                        attr = name
                    setattr(
                        submodule,
                        attr,
                        _make_tensor_pack_like(x, data=x.data[pack_idx]),
                    )
                    original_modules_and_tensors.append((submodule, attr, x))

            for submodule in self.modules():
                if isinstance(submodule, PackView):
                    original_pack_views.append((submodule, submodule.pack_size))
                    submodule._pack_size = len(pack_idx)

            yield

        finally:
            for submodule, attr, parameter in original_modules_and_tensors:
                setattr(submodule, attr, parameter)
            for submodule, pack_size in original_pack_views:
                submodule._pack_size = pack_size


def _check_module_pack(module: ModulePack) -> None:
    for word, cls, iter_named_tensors in (
        ('parameter', ParameterPack, module.named_parameters),
        ('buffer', BufferPack, module.named_buffers),
    ):
        values_to_names = {}
        for name, value in iter_named_tensors(remove_duplicate=False):
            if value in values_to_names:
                raise RuntimeError(
                    f'The module pack contains aliased {word}s, which is not allowed:'
                    f' "{values_to_names[value]}" and "{name}"'  # ty: ignore[invalid-argument-type]
                )
            if not isinstance(value, cls):
                raise RuntimeError(  # noqa: TRY004
                    f'In module packs, {cls} must be used to store {word}s.'
                    f' However, the {word} "{name}" is stored as `{type(value)}`.'
                )
            values_to_names[value] = name


def _prepare_positive_integer_pack(
    value: int | list[int], max_value: int | None, *, pack_size: int, **factory_kwargs
) -> tuple[BufferPack | None, int]:
    if isinstance(value, list):
        # The values are specified for all pack members separately.
        if len(value) != pack_size:
            raise ValueError(
                'The number of per-member hyperparameter values does not match the pack'
                ' size'
            )
        if not all(x > 0 for x in value):
            raise ValueError('All values must be positive')

        if max_value is None:
            max_value = max(value)
        elif not all(x <= max_value for x in value):
            raise ValueError('Values must not exceed `max_value`')

        value_buffer = BufferPack(torch.tensor(value, **factory_kwargs))
    else:
        # The same value for all pack members.
        if value <= 0:
            raise ValueError('The value must be positive')

        if max_value is None:
            max_value = value
        elif max_value != value:
            raise ValueError('The configured maximum must equal the shared value')

        value_buffer = None

    return value_buffer, max_value


def _check_replace_arguments(
    pack_idx: Tensor, *hyperparameter_lists: list | None
) -> None:
    if not all(x is None or len(x) == len(pack_idx) for x in hyperparameter_lists):
        raise ValueError(
            'Each provided hyperparameter list must contain one value for each selected'
            ' pack member'
        )


class LinearPack(ModulePack):
    """A pack of linear layers.

    **Shape**

    - Input: `(pack_size, batch_size, self.max_in_features)`
    - Output: `(pack_size, batch_size, self.max_out_features)`

    **Implementation details**

    * The module assumes that the input is already properly masked according to
      `in_features` passed to the constructor. Therefore, `in_features` is only used for
      the parameter initialization in `reset_parameters`, but not in `forward`.
    * The module properly masks its output in `forward` according to `out_features`
      passed to the constructor.

    **Examples**

    The same layer shapes for all pack members:

    >>> m = LinearPack(4, 5, pack_size=2)
    >>> m.pack_size
    2
    >>> x = torch.randn(2, 3, 4)
    >>> m(x).shape  # Input masking is assumed
    torch.Size([2, 3, 5])

    Different layer shapes:

    >>> m = LinearPack([4, 5], [6, 7], pack_size=2)
    >>> m.max_in_features
    5
    >>> m.max_out_features
    7
    >>> x = torch.randn(2, 3, 5)
    >>> m(x).shape  # Input masking is assumed
    torch.Size([2, 3, 7])

    Different layer shapes, with spare capacity for replacing layers with larger ones
    later:

    >>> m = LinearPack(
    ...     [4, 5], [6, 7], max_in_features=8, max_out_features=9, pack_size=2
    ... )
    >>> m.max_in_features
    8
    >>> m.max_out_features
    9
    >>> x = torch.randn(2, 3, 8)
    >>> m(x).shape  # Input masking is assumed
    torch.Size([2, 3, 9])
    """

    def __init__(
        self,
        in_features: int | list[int],
        out_features: int | list[int],
        bias: bool = True,
        *,
        max_in_features: int | None = None,
        max_out_features: int | None = None,
        pack_size: int,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if pack_size <= 0:
            raise ValueError('The pack size must be positive')

        factory_kwargs: KWArgs = {'dtype': dtype, 'device': device}
        in_features_buffer, max_in_features = _prepare_positive_integer_pack(
            in_features, max_in_features, pack_size=pack_size, **factory_kwargs
        )
        out_features_buffer, max_out_features = _prepare_positive_integer_pack(
            out_features, max_out_features, pack_size=pack_size, **factory_kwargs
        )

        super().__init__(pack_size=pack_size)
        self.in_features = in_features_buffer
        self.out_features = out_features_buffer
        self.weight = ParameterPack(
            torch.empty(pack_size, max_out_features, max_in_features, **factory_kwargs)
        )
        self.bias = (
            ParameterPack(torch.empty(pack_size, max_out_features, **factory_kwargs))
            if bias
            else None
        )

        self.reset_parameters()

    @property
    def max_in_features(self) -> int:
        """The maximum supported number of input features."""
        return self.weight.shape[-1]

    @property
    def max_out_features(self) -> int:
        """The maximum supported number of output features."""
        return self.weight.shape[-2]

    @property
    def pack_size(self) -> int:
        return self.weight.shape[PACK_DIM]

    def reset_parameters(self, *, pack_idx: Tensor | None = None) -> None:
        """Initialize the parameters."""
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        if self.in_features is None:
            d_in_rsqrt = self.max_in_features**-0.5
            init_args = (-d_in_rsqrt, d_in_rsqrt)

            for p in (self.weight, self.bias):
                if p is not None:
                    if pack_idx is None:
                        nn.init.uniform_(p, *init_args)
                    else:
                        with torch.no_grad():
                            p[pack_idx] = torch.empty(
                                len(pack_idx),
                                *p.shape[1:],
                                dtype=p.dtype,
                                device=p.device,
                            ).uniform_(*init_args)

        else:
            in_features = _get_tensor_pack(self.in_features, pack_idx)
            d_in_rsqrt = in_features.float().rsqrt()
            d_in_rsqrt_neg = -d_in_rsqrt

            for p in (self.weight, self.bias):
                if p is not None:
                    single_shape = p.shape[1:]
                    broadcast_idx = (slice(None), *((None,) * (p.ndim - 1)))
                    p_init = torch.lerp(
                        d_in_rsqrt_neg[broadcast_idx].expand(-1, *single_shape),
                        d_in_rsqrt[broadcast_idx].expand(-1, *single_shape),
                        torch.rand(
                            self.pack_size if pack_idx is None else len(pack_idx),
                            *single_shape,
                            dtype=p.dtype,
                            device=p.device,
                        ),
                    )
                    with torch.no_grad():
                        if pack_idx is None:
                            p.copy_(p_init)
                        else:
                            p[pack_idx] = p_init

    def replace(
        self,
        *,
        pack_idx: Tensor,
        in_features: list[int] | None = None,
        out_features: list[int] | None = None,
    ) -> None:
        check_pack_idx(pack_idx, self.pack_size)
        _check_replace_arguments(pack_idx, in_features, out_features)

        updates = []
        for buffer, max_n_features, n_features in [
            (self.in_features, self.max_in_features, in_features),
            (self.out_features, self.max_out_features, out_features),
        ]:
            if n_features is not None:
                if buffer is None:
                    raise RuntimeError(
                        'Cannot replace feature counts that are shared by all pack'
                        ' members'
                    )
                if not all(0 < x <= max_n_features for x in n_features):
                    raise ValueError(
                        'Feature counts must be positive and must not exceed the'
                        ' allocated maximum'
                    )

                # Validate both feature counts before updating either buffer.
                updates.append(
                    (
                        buffer,
                        torch.tensor(
                            n_features, dtype=buffer.dtype, device=buffer.device
                        ),
                    )
                )
        for buffer, values in updates:
            buffer[pack_idx] = values

        # The parameters are reset _after_ updating the hyperparameters, because the
        # initialization depends on the hyperparameter values.
        self.reset_parameters(pack_idx=pack_idx)

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        if x.shape[-1] != self.max_in_features:
            raise ValueError(
                'The number of input features must be equal to the maximum supported'
                ' number of features'
            )

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        out_features = _get_tensor_pack(self.out_features, pack_idx)

        x = (
            torch.bmm(x, weight.transpose(-2, -1))
            if bias is None
            else torch.baddbmm(bias.unsqueeze(BATCH_DIM), x, weight.transpose(-2, -1))
        )
        if out_features is not None:
            output_mask = (
                torch.arange(self.max_out_features, device=x.device)[None]
                < out_features[:, None]
            )
            x = x * output_mask.to(x.dtype).unsqueeze(BATCH_DIM)
        return x


class DropoutPack(ModulePack):
    """A pack of dropout layers.

    Dropout probabilities must be in `[0, 1)`.

    **Shapes**

    - Input: `(pack_size, *)`
    - Output: `(pack_size, *)`

    **Examples**

    The same dropout ratio for all pack members:

    >>> m = DropoutPack(0.5, pack_size=2)
    >>> x = torch.randn(2, 3, 4)
    >>> m(x).shape
    torch.Size([2, 3, 4])

    Different dropout ratios:

    >>> m = DropoutPack([0.1, 0.5], pack_size=2)
    >>> x = torch.randn(2, 3, 4)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, p: float | list[float], *, pack_size: int) -> None:
        if isinstance(p, int | float):
            if not math.isfinite(p) or p < 0.0 or p >= 1.0:
                raise ValueError('The dropout probability must be finite and in [0, 1)')
        else:
            if len(p) != pack_size:
                raise ValueError('Provide one dropout probability per pack member')
            if any(not math.isfinite(x) or x < 0.0 or x >= 1.0 for x in p):
                raise ValueError('Dropout probabilities must be finite and in [0, 1)')

        super().__init__(pack_size=pack_size)
        self.p = (
            float(p)
            if isinstance(p, int | float)
            else BufferPack(torch.tensor([float(x) for x in p]))
        )

    @property
    def pack_size(self) -> int:
        if isinstance(self.p, int | float):
            raise RuntimeError('Cannot infer the pack size')  # noqa: TRY004
        return self.p.shape[PACK_DIM]

    def replace(self, *, pack_idx: Tensor, p: list[float] | None = None) -> None:
        if p is None:
            return

        if not isinstance(self.p, BufferPack):
            raise RuntimeError(  # noqa: TRY004
                'Cannot replace dropout probabilities when one probability is shared by'
                ' all pack members'
            )

        check_pack_idx(pack_idx, self.pack_size)
        _check_replace_arguments(pack_idx, p)
        if any(not math.isfinite(x) or x < 0.0 or x >= 1.0 for x in p):
            raise ValueError('Dropout probabilities must be finite and in [0, 1)')

        self.p[pack_idx] = torch.tensor(p, dtype=self.p.dtype, device=self.p.device)

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.try_get_pack_size())

        # Passing the tuple instead of `int | float` keeps code traceable by
        # PyTorch 2.7-2.9 strict export.
        if isinstance(self.p, (int, float)):
            return nn.functional.dropout(x, p=self.p, training=self.training)

        p = _get_tensor_pack(self.p, pack_idx)

        if self.training:
            p_keep = 1.0 - p
            p_keep = p_keep[(slice(None), *((None,) * (x.ndim - 1)))]
            mask = torch.empty_like(x).bernoulli_(p_keep.expand_as(x))
            return mask.div_(p_keep) * x
        else:
            return x


class LeakyReLUPack(ModulePack):
    """A pack of leaky ReLU activations.

    This class is not used in the original paper experiments and exists as an additional
    illustration of how the "Module Pack" pattern can be used.

    **Shapes**

    - Input: `(pack_size, *)`
    - Output: `(pack_size, *)`

    **Examples**

    >>> m = LeakyReLUPack(negative_slope=[0.01, 0.02], pack_size=2)
    >>> x = torch.randn(2, 3)
    >>> m(x).shape
    torch.Size([2, 3])
    """

    def __init__(self, *, negative_slope: float | list[float], pack_size: int) -> None:
        if isinstance(negative_slope, int | float):
            if not math.isfinite(negative_slope) or negative_slope <= 0:
                raise ValueError('The slope must be finite and positive')
        else:
            if len(negative_slope) != pack_size:
                raise ValueError(
                    'The size of the list of per-member slope values must match the'
                    ' pack size'
                )
            if any(not math.isfinite(x) or x <= 0 for x in negative_slope):
                raise ValueError('Slopes must be finite and positive')

        super().__init__(pack_size=pack_size)
        self.negative_slope = (
            BufferPack(torch.tensor([float(x) for x in negative_slope]))
            if isinstance(negative_slope, list)
            else float(negative_slope)
        )

    @property
    def pack_size(self) -> int:
        if isinstance(self.negative_slope, int | float):
            raise RuntimeError('Cannot infer the pack size')  # noqa: TRY004
        return self.negative_slope.shape[PACK_DIM]

    def replace(
        self, *, pack_idx: Tensor, negative_slope: list[float] | None = None
    ) -> None:
        if negative_slope is None:
            return

        if not isinstance(self.negative_slope, BufferPack):
            raise RuntimeError(  # noqa: TRY004
                'Cannot replace slopes that are shared by all pack members'
            )

        check_pack_idx(pack_idx, self.pack_size)
        _check_replace_arguments(pack_idx, negative_slope)
        if any(not math.isfinite(x) or x <= 0 for x in negative_slope):
            raise ValueError('Slopes must be finite and positive')

        self.negative_slope[pack_idx] = torch.tensor(
            negative_slope,
            dtype=self.negative_slope.dtype,
            device=self.negative_slope.device,
        )

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.try_get_pack_size())

        # Passing the tuple instead of `int | float` keeps code traceable by
        # PyTorch 2.7-2.9 strict export.
        if isinstance(self.negative_slope, (int, float)):
            return nn.functional.leaky_relu(x, self.negative_slope)

        negative_slope = _get_tensor_pack(self.negative_slope, pack_idx)
        unsqueeze_idx = (slice(None), *((None,) * (x.ndim - 1)))
        return torch.where(x >= 0, x, x * negative_slope[unsqueeze_idx])


def _make_activation(type: type[nn.Module], *, pack_size: int, **kwargs) -> nn.Module:
    cls = type
    del type

    return cls(
        **kwargs, **({'pack_size': pack_size} if issubclass(cls, ModulePack) else {})
    )


class MLPBackbonePack(ModulePack):
    """A pack of multilayer perceptron backbones.

    **Examples**

    >>> m = MLPBackbonePack(
    ...     d_in=[4, 5],
    ...     n_blocks=[6, 7],
    ...     d_block=[8, 9],
    ...     dropout=[0.1, 0.2],
    ...     pack_size=2,
    ... )
    >>> m.pack_size
    2
    >>> x = torch.randn(2, 3, 5)
    >>> m(x).shape  # Input masking is assumed
    torch.Size([2, 3, 9])
    """

    class Block(ModulePack):
        """A pack of MLP blocks (linear layer + activation + dropout)."""

        def __init__(
            self,
            *,
            activation: KWArgs,
            dropout: float | list[float],
            pack_size: int,
            **linear_kwargs,
        ) -> None:
            super().__init__(pack_size=pack_size)
            self.linear = LinearPack(pack_size=pack_size, **linear_kwargs)
            self.activation = _make_activation(pack_size=pack_size, **activation)
            self.dropout = DropoutPack(dropout, pack_size=pack_size)

        @property
        def pack_size(self) -> int:
            return self.linear.pack_size

        def replace(
            self,
            *,
            pack_idx: Tensor,
            in_features: list[int] | None = None,
            out_features: list[int] | None = None,
            activation: KWArgs | None = None,
            dropout: list[float] | None = None,
        ) -> None:
            self.linear.replace(
                pack_idx=pack_idx, in_features=in_features, out_features=out_features
            )

            activation_replace = (
                getattr(self.activation, 'replace', None)
                if isinstance(self.activation, ModulePack)
                else None
            )
            if activation_replace is not None and not inspect.ismethod(
                activation_replace
            ):
                raise RuntimeError(
                    'The activation module does not support member replacement'
                )
            if activation is not None and activation_replace is None:
                raise RuntimeError(
                    'The activation module does not support member replacement'
                )

            if activation_replace is not None:
                activation_replace(
                    pack_idx=pack_idx, **({} if activation is None else activation)
                )

            self.dropout.replace(pack_idx=pack_idx, p=dropout)

        def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
            x = self.linear(x, pack_idx)
            x = (
                self.activation(x, pack_idx)
                if isinstance(self.activation, ModulePack)
                else self.activation(x)
            )
            x = self.dropout(x, pack_idx)
            return x

    @enum.unique
    class ForwardImpl(enum.StrEnum):
        """The implementation of the forward pass."""

        DENSE = enum.auto()
        """
        All blocks are applied to all pack members with masking. This implementation is
        usually faster for light workloads (e.g., small input batches) and makes the
        module easier to export.
        """

        INDEXED = enum.auto()
        """
        In each block, only the pack members that are still active are computed. This
        implementation is faster for training (large) packs of depth-varying MLPs. For
        non-trivial export scenarios (e.g., using ONNX or `torch.export`) with this
        implementation:

        - First, set the module's device and data type as needed.
        - Then, do a warmup forward pass _without_ passing `pack_idx`.
        - Then, export the module. The exported module must be used _without_ passing
          `pack_idx` to the forward pass.

        Note:
            With this implementation, for depth-varying packs, passing `pack_idx` to the
            forward pass triggers a host-device synchronization in addition to the usual
            overhead associated with using `pack_idx`. See the documentation for
            `ModulePack` for more details on the `pack_idx` mechanism.
        """

    class _ActiveLocalPackIdxCache(NamedTuple):
        n_blocks: BufferPack | None
        n_blocks_version: int | None
        value: list[Tensor | None]

    @staticmethod
    def _default_load_state_dict_post_hook(m: 'MLPBackbonePack', _) -> None:
        m._reset_active_local_pack_idx_cache()

    def __init__(
        self,
        *,
        d_in: int | list[int],
        max_d_in: int | None = None,
        n_blocks: int | list[int],
        d_block: int | list[int],
        activation: KWArgs | None = None,
        dropout: float | list[float],
        max_n_blocks: int | None = None,
        max_d_block: int | None = None,
        forward_impl: ForwardImpl | str = ForwardImpl.DENSE,
        pack_size: int,
    ) -> None:
        n_blocks_buffer, max_n_blocks = _prepare_positive_integer_pack(
            n_blocks, max_n_blocks, pack_size=pack_size
        )

        super().__init__(pack_size=pack_size)
        cls = type(self)
        self.blocks = nn.ModuleList(
            [
                cls.Block(
                    in_features=d_in if i == 0 else d_block,
                    out_features=d_block,
                    max_in_features=max_d_in if i == 0 else max_d_block,
                    max_out_features=max_d_block,
                    activation={'type': nn.ReLU} if activation is None else activation,
                    dropout=dropout,
                    pack_size=pack_size,
                )
                for i in range(max_n_blocks)
            ]
        )

        self._max_n_blocks: Final = max_n_blocks
        self.n_blocks = n_blocks_buffer
        # For depth-varying packs, deeper blocks may have fewer active members. Their
        # indices are computed from `self.n_blocks` and cached to avoid repeating the
        # host-device synchronization required to compute them.
        #
        # Keep the cached `self.n_blocks` reference inside the tuple: assigning it
        # directly as a module attribute would register an aliased buffer, which
        # is forbidden in module packs.
        #
        # Finally, note that modifying `self.n_blocks` in place invalidates the cache.
        # Therefore, use `self.n_blocks_index_put_` for any in-place modifications.
        self._active_local_pack_idx_cache: (
            MLPBackbonePack._ActiveLocalPackIdxCache | None
        ) = None

        self.register_load_state_dict_post_hook(
            MLPBackbonePack._default_load_state_dict_post_hook
        )

        self._forward_impl = MLPBackbonePack.ForwardImpl(forward_impl)

    @property
    def pack_size(self) -> int:
        return self.blocks[0].pack_size  # ty: ignore[invalid-return-type]

    @property
    def max_n_blocks(self) -> int:
        """The maximum supported number of blocks."""
        return self._max_n_blocks

    @property
    def forward_impl(self) -> ForwardImpl:
        """The implementation of the forward pass."""
        return self._forward_impl

    def set_forward_impl(self, impl: ForwardImpl | str) -> None:
        """Set the implementation of the forward pass."""
        self._forward_impl = MLPBackbonePack.ForwardImpl(impl)

    def _reset_active_local_pack_idx_cache(self) -> None:
        self._active_local_pack_idx_cache = None

    def _load_from_state_dict_at(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        *,
        pack_idx: Tensor,
        state_dict_idx: Tensor | None,
    ) -> None:
        self._reset_active_local_pack_idx_cache()
        super()._load_from_state_dict_at(
            state_dict, prefix, pack_idx=pack_idx, state_dict_idx=state_dict_idx
        )

    def n_blocks_index_put_(self, indices: tuple[Tensor, ...], values: Tensor) -> None:
        """Modify `n_blocks` in place.

        This method is the only correct way to modify `n_blocks` in place. The method
        directly calls `self.n_blocks.index_put_` with the provided arguments, and
        updates the parts of the module's state that depend on `self.n_blocks`.
        """
        if self.n_blocks is None:
            raise RuntimeError(
                'Cannot change block counts when all pack members share a fixed depth'
            )

        self._reset_active_local_pack_idx_cache()
        self.n_blocks.index_put_(indices, values)

    def replace(
        self,
        *,
        pack_idx: Tensor,
        d_in: list[int] | None = None,
        n_blocks: list[int] | None = None,
        d_block: list[int] | None = None,
        **kwargs,
    ) -> None:
        """
        All blocks of a given MLP share the replacement width, activation settings,
        and dropout rate.
        """
        check_pack_idx(pack_idx, self.pack_size)
        _check_replace_arguments(pack_idx, n_blocks)

        if n_blocks is None:
            new_n_blocks = None
        else:
            if self.n_blocks is None:
                raise RuntimeError(
                    'Cannot change block counts when all pack members share a fixed'
                    ' depth'
                )
            if not all(0 < x <= self.max_n_blocks for x in n_blocks):
                raise ValueError(
                    'Block counts must be positive and must not exceed `max_n_blocks`'
                )

            # Defer the depth update until all block replacements succeed.
            new_n_blocks = torch.tensor(
                n_blocks, dtype=self.n_blocks.dtype, device=self.n_blocks.device
            )

        # For simplicity, all the blocks are replaced regardless of `n_blocks`.
        for i_block, block in enumerate(self.blocks):
            block.replace(
                pack_idx=pack_idx,
                in_features=d_in if i_block == 0 else d_block,
                out_features=d_block,
                **kwargs,
            )

        if new_n_blocks is not None:
            self.n_blocks_index_put_((pack_idx,), new_n_blocks)

    def _forward_dense(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        n_blocks = _get_tensor_pack(self.n_blocks, pack_idx)
        for i_block, block in enumerate(self.blocks):
            x_new = block(x, pack_idx)
            # All members are active in the first block.
            if n_blocks is None or i_block == 0:
                x = x_new
            else:
                assert PACK_DIM == 0
                active_pack_mask = i_block < n_blocks
                x = torch.where(
                    active_pack_mask.view(-1, *((1,) * (x.ndim - 1))), x_new, x
                )
        return x

    # The method must return non-inference tensors to make the computed indices usable
    # in all modes regardless of the mode in which the indices are computed.
    @torch.inference_mode(False)
    def _compute_active_local_pack_idx_by_block(
        self, pack_idx: Tensor | None
    ) -> list[Tensor | None]:
        n_blocks = _get_tensor_pack(self.n_blocks, pack_idx)

        if n_blocks is None:
            return [None for _ in range(self.max_n_blocks)]

        else:
            pack_size = self.pack_size if pack_idx is None else len(pack_idx)
            # Consider the following example (`pack_size = 6`):
            # n_blocks = [1, 2, 3, 1, 2, 3]
            #
            # The i-th row of `mask` indicates which pack members are active
            # in the i-th block:
            # mask = [
            #     [True,  True,  True, True,  True,  True],
            #     [False, True,  True, False, True,  True],
            #     [False, False, True, False, False, True],
            # ]
            #
            # `counts` contains the number of pack members active in each block:
            # counts = [6, 4, 2]
            #
            # `counts` enables the use of `nonzero_static` instead of `nonzero`,
            # reducing the number of host-device synchronizations in this function
            # to one (triggered by the `.tolist()` call below).
            mask = (
                torch.arange(self.max_n_blocks, device=n_blocks.device)[:, None]
                < n_blocks[None]
            )
            counts = mask.long().sum(1).tolist()
            return [
                (
                    # `None` means that all selected members have a depth of at least
                    # `i_block + 1`.
                    None
                    if counts[i_block] == pack_size
                    else torch.nonzero_static(mask[i_block], size=counts[i_block])[
                        :, 0
                    ].clone()
                )
                for i_block in range(self.max_n_blocks)
            ]

    def _get_active_local_pack_idx_by_block(
        self, pack_idx: Tensor | None
    ) -> list[Tensor | None]:
        if pack_idx is not None:
            # The cache is maintained only for the whole pack, since caching the result
            # for arbitrary `pack_idx` values would require a significantly more complex
            # caching mechanism. As a result, in depth-varying packs, member indexing
            # always triggers the host-device synchronization described in the index
            # computation method.
            return self._compute_active_local_pack_idx_by_block(pack_idx)

        n_blocks_version = (
            None
            if self.n_blocks is None or self.n_blocks.is_inference()
            else self.n_blocks._version
        )
        if (
            self._active_local_pack_idx_cache is None
            or self._active_local_pack_idx_cache.n_blocks is not self.n_blocks
            # Additional protection from in-place modifications of `n_blocks` without
            # the `n_blocks_index_put_` method. The protection is not absolute, in
            # particular it does not detect modifications of inference tensors (which
            # have no version counter) or modifications through `n_blocks.data`.
            # Any code outside of this method must _not_ rely on this additional
            # protection and should be written as this protection does not exist.
            or self._active_local_pack_idx_cache.n_blocks_version != n_blocks_version
        ):
            self._reset_active_local_pack_idx_cache()
            self._active_local_pack_idx_cache = (
                MLPBackbonePack._ActiveLocalPackIdxCache(
                    self.n_blocks,
                    n_blocks_version,
                    self._compute_active_local_pack_idx_by_block(None),
                )
            )
        return self._active_local_pack_idx_cache.value

    def _forward_indexed(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        # Without autograd, the in-place version is safe and faster. It never modifies
        # the input, because all members are active in the first block, which is
        # explicitly checked by the assertion below.
        index_copy = (
            Tensor.index_copy if torch.is_grad_enabled() else Tensor.index_copy_
        )

        active_local_pack_idx_by_block = self._get_active_local_pack_idx_by_block(
            pack_idx
        )
        assert (
            active_local_pack_idx_by_block and active_local_pack_idx_by_block[0] is None
        )

        for block, active_local_pack_idx in zip(
            self.blocks, active_local_pack_idx_by_block, strict=True
        ):
            if active_local_pack_idx is None:
                # All the pack members continue the forward pass.
                x = block(x, pack_idx)
            elif active_local_pack_idx.numel() > 0:
                # Only some of the pack members continue the forward pass. Note that
                # the computed indices are applied to `x`, i.e. they refer to the
                # selected members, while the block itself must be indexed with the
                # corresponding indices in the whole pack.
                x = index_copy(
                    x,
                    PACK_DIM,
                    active_local_pack_idx,
                    block(
                        x.index_select(PACK_DIM, active_local_pack_idx),
                        (
                            active_local_pack_idx
                            if pack_idx is None
                            else pack_idx[active_local_pack_idx]
                        ),
                    ),
                )
            else:
                # None of the pack members continue the forward pass.
                break
        return x

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if self._forward_impl is MLPBackbonePack.ForwardImpl.INDEXED:
            return self._forward_indexed(x, pack_idx)
        elif self._forward_impl is MLPBackbonePack.ForwardImpl.DENSE:
            return self._forward_dense(x, pack_idx)
        else:
            raise ValueError(f'Unknown forward implementation: {self._forward_impl}')


class NumericalFeatureEmbeddingsPack(ModulePack, abc.ABC):
    """The base class for packs of numerical feature embeddings.

    Each numerical feature is mapped to its own learned vector before entering the main
    backbone. `d_embedding` controls the vector size per feature.

    In addition to implementing the abstract methods, inherited classes must also
    accept the following arguments in their constructors:

    - `n_features`
    - `d_embedding`
    - `max_d_embedding`
    - `pack_size`
    """

    @abc.abstractmethod
    def get_output_shape(self) -> torch.Size:
        """Get the shape of the embeddings of one object."""
        ...


class LinearEmbeddingsPack(NumericalFeatureEmbeddingsPack):
    """A pack of linear feature embeddings."""

    def __init__(
        self,
        n_features: int,
        d_embedding: int | list[int],
        *,
        max_d_embedding: int | None = None,
        pack_size: int,
    ) -> None:
        d_embedding_buffer, max_d_embedding = _prepare_positive_integer_pack(
            d_embedding, max_d_embedding, pack_size=pack_size
        )

        super().__init__(pack_size=pack_size)
        self.weight = ParameterPack(torch.empty(pack_size, n_features, max_d_embedding))
        self.bias = ParameterPack(torch.empty(pack_size, n_features, max_d_embedding))
        self.d_embedding = d_embedding_buffer
        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return self.weight.shape[PACK_DIM]

    @property
    def max_d_embedding(self) -> int:
        """The maximum supported embedding size."""
        return self.weight.shape[-1]

    def get_output_shape(self) -> torch.Size:
        """Get the shape of the embeddings of one object."""
        return self.weight.shape[1:]  # ty:ignore[invalid-return-type]

    def reset_parameters(self, *, pack_idx: Tensor | None = None) -> None:
        """Initialize the parameters."""
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        if self.d_embedding is None:
            d_rsqrt = self.max_d_embedding**-0.5
            init_args = (-d_rsqrt, d_rsqrt)

            for p in (self.weight, self.bias):
                if pack_idx is None:
                    nn.init.uniform_(p, *init_args)
                else:
                    with torch.no_grad():
                        p[pack_idx] = torch.empty(
                            len(pack_idx), *p.shape[1:], dtype=p.dtype, device=p.device
                        ).uniform_(*init_args)

        else:
            d_embedding = _get_tensor_pack(self.d_embedding, pack_idx)
            d_rsqrt = d_embedding.float().rsqrt()
            d_rsqrt_neg = -d_rsqrt

            for p in (self.weight, self.bias):
                single_shape = p.shape[1:]
                broadcast_idx = (slice(None), *((None,) * (p.ndim - 1)))
                p_init = torch.lerp(
                    d_rsqrt_neg[broadcast_idx].expand(-1, *single_shape),
                    d_rsqrt[broadcast_idx].expand(-1, *single_shape),
                    torch.rand(
                        self.pack_size if pack_idx is None else len(pack_idx),
                        *single_shape,
                        dtype=p.dtype,
                        device=p.device,
                    ),
                )
                with torch.no_grad():
                    if pack_idx is None:
                        p.copy_(p_init)
                    else:
                        p[pack_idx] = p_init

    def replace(
        self, *, pack_idx: Tensor, d_embedding: list[int] | None = None
    ) -> None:
        check_pack_idx(pack_idx, self.pack_size)
        _check_replace_arguments(pack_idx, d_embedding)

        if d_embedding is not None:
            if not isinstance(self.d_embedding, BufferPack):
                raise RuntimeError(
                    'Cannot replace embedding sizes that are shared by all pack members'
                )
            if not all(0 < x <= self.max_d_embedding for x in d_embedding):
                raise ValueError(
                    'Embedding sizes must be positive and must not exceed'
                    ' `max_d_embedding`'
                )

            self.d_embedding[pack_idx] = torch.tensor(
                d_embedding,
                dtype=self.d_embedding.dtype,
                device=self.d_embedding.device,
            )

        # The parameters are reset _after_ updating the hyperparameters, because the
        # initialization depends on the hyperparameter values.
        self.reset_parameters(pack_idx=pack_idx)

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        if x.shape[-1] != self.weight.shape[-2]:
            raise ValueError(
                'The number of input features must match the number of features'
                ' configured for the embeddings'
            )
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        d_embedding = _get_tensor_pack(self.d_embedding, pack_idx)

        if d_embedding is not None:
            # Masking parameters is significantly faster than masking activations.
            parameter_mask = (
                torch.arange(self.max_d_embedding, device=x.device)[None]
                < d_embedding[:, None]
            )
            parameter_mask = parameter_mask.float().unsqueeze(-2)
            weight = weight * parameter_mask
            bias = bias * parameter_mask

        return torch.addcmul(
            bias.unsqueeze(BATCH_DIM), weight.unsqueeze(BATCH_DIM), x[..., None]
        )


class LinearReLUEmbeddingsPack(NumericalFeatureEmbeddingsPack):
    """A pack of linear embeddings followed by the ReLU activation."""

    def __init__(
        self,
        n_features: int,
        d_embedding: int | list[int],
        *,
        max_d_embedding: int | None = None,
        pack_size: int,
    ) -> None:
        super().__init__(pack_size=pack_size)
        self.linear = LinearEmbeddingsPack(
            n_features,
            d_embedding,
            max_d_embedding=max_d_embedding,
            pack_size=pack_size,
        )
        self.activation = nn.ReLU()

    @property
    def pack_size(self) -> int:
        return self.linear.pack_size

    @property
    def max_d_embedding(self) -> int:
        """The maximum supported embedding size."""
        return self.linear.max_d_embedding

    def get_output_shape(self) -> torch.Size:
        """Get the shape of the embeddings of one object."""
        return self.linear.get_output_shape()

    def replace(
        self, *, pack_idx: Tensor, d_embedding: list[int] | None = None
    ) -> None:
        self.linear.replace(pack_idx=pack_idx, d_embedding=d_embedding)

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        return self.activation(self.linear(x, pack_idx))


class CosineEmbeddingsPack(NumericalFeatureEmbeddingsPack):
    """A pack of cosine feature embeddings.

    These embeddings are proposed in the TabPack paper. They combine learned cosine
    functions of each numerical feature with the original feature value. `init_scale`
    controls the initial cosine frequencies.

    **Shape**

    - Input: `(pack_size, batch_size, n_features)`
    - Output: `(pack_size, batch_size, n_features, self.max_d_embedding)`

    **Examples**

    >>> m = CosineEmbeddingsPack(4, [5, 6], init_scale=[0.01, 0.1], pack_size=2)
    >>> m.max_d_embedding
    6
    >>> x = torch.randn(2, 3, 4)
    >>> m(x).shape
    torch.Size([2, 3, 4, 6])
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int | list[int],
        *,
        max_d_embedding: int | None = None,
        init_scale: float | list[float],
        pack_size: int,
    ) -> None:
        concat_input = True

        d_embedding_buffer, max_d_embedding = _prepare_positive_integer_pack(
            d_embedding, max_d_embedding, pack_size=pack_size
        )

        super().__init__(pack_size=pack_size)
        self.weight = ParameterPack(
            torch.empty(pack_size, n_features, max_d_embedding - int(concat_input))
        )
        self.bias = ParameterPack(torch.empty(self.weight.shape))
        self.elementwise_affine_weight = ParameterPack(
            torch.empty(
                *self.weight.shape[:-1], self.weight.shape[-1] + int(concat_input)
            )
        )
        self.elementwise_affine_bias = ParameterPack(
            torch.empty(self.elementwise_affine_weight.shape)
        )

        self.d_embedding = d_embedding_buffer
        self.init_scale = (
            float(init_scale)
            if isinstance(init_scale, int | float)
            else BufferPack(torch.tensor([float(x) for x in init_scale]))
        )

        self._concat_input: Final = concat_input

        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return self.weight.shape[PACK_DIM]

    @property
    def max_d_embedding(self) -> int:
        """The maximum supported embedding size."""
        return self.weight.shape[-1] + int(self._concat_input)

    def get_output_shape(self) -> torch.Size:
        """Get the shape of the embeddings of one object."""
        return torch.Size(
            (self.weight.shape[1], self.weight.shape[2] + int(self._concat_input))
        )

    def reset_parameters(self, *, pack_idx: Tensor | None = None) -> None:
        """Initialize the parameters."""
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        bound = 3.0
        weight = (
            self.weight
            if pack_idx is None
            else torch.empty(
                len(pack_idx),
                *self.weight.shape[1:],
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
        )
        nn.init.trunc_normal_(weight, a=-bound, b=bound)
        with torch.no_grad():
            if isinstance(self.init_scale, int | float):
                weight *= self.init_scale
            else:
                init_scale = _get_tensor_pack(self.init_scale, pack_idx)
                weight *= init_scale[(slice(None), *((None,) * (weight.ndim - 1)))]
        if pack_idx is not None:
            with torch.no_grad():
                self.weight[pack_idx] = weight

        for p in (
            self.bias,
            self.elementwise_affine_weight,
            self.elementwise_affine_bias,
        ):
            if pack_idx is None:
                nn.init.zeros_(p)
            else:
                with torch.no_grad():
                    p[pack_idx] = 0.0

    def replace(
        self,
        *,
        pack_idx: Tensor,
        d_embedding: list[int] | None = None,
        init_scale: list[float] | None = None,
    ) -> None:
        check_pack_idx(pack_idx, self.pack_size)
        _check_replace_arguments(pack_idx, d_embedding, init_scale)

        updates = []
        if d_embedding is not None:
            if not isinstance(self.d_embedding, BufferPack):
                raise RuntimeError(
                    'Cannot replace embedding sizes that are shared by all pack members'
                )
            if not all(0 < x <= self.max_d_embedding for x in d_embedding):
                raise ValueError(
                    'Embedding sizes must be positive and must not exceed'
                    ' `max_d_embedding`'
                )

            # Defer buffer updates until both hyperparameters have been checked.
            updates.append((self.d_embedding, d_embedding))
        if init_scale is not None:
            if not isinstance(self.init_scale, BufferPack):
                raise RuntimeError(
                    'Cannot replace initialization scales that are shared by all pack'
                    ' members'
                )

            updates.append((self.init_scale, init_scale))
        for buffer, values in updates:
            buffer[pack_idx] = torch.tensor(
                values, dtype=buffer.dtype, device=buffer.device
            )

        # The parameters are reset _after_ updating the hyperparameters, because the
        # initialization depends on the hyperparameter values.
        self.reset_parameters(pack_idx=pack_idx)

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        if x.shape[-1] != self.weight.shape[-2]:
            raise ValueError(
                'The number of input features must match the number of features'
                ' configured for the embeddings'
            )
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        elementwise_affine_weight = _get_tensor_pack(
            self.elementwise_affine_weight, pack_idx
        )
        elementwise_affine_bias = _get_tensor_pack(
            self.elementwise_affine_bias, pack_idx
        )
        d_embedding = _get_tensor_pack(self.d_embedding, pack_idx)

        if d_embedding is not None:
            # Masking parameters is significantly faster than masking activations.
            parameter_mask = (
                torch.arange(self.max_d_embedding, device=x.device)[None]
                < d_embedding[:, None]
            )
            parameter_mask = parameter_mask.float().unsqueeze(-2)
            elementwise_affine_weight = elementwise_affine_weight * parameter_mask
            elementwise_affine_bias = elementwise_affine_bias * parameter_mask

        # Scaling parameters is usually faster than scaling the input, especially for
        # non-trivial batch sizes.
        weight = weight * (2 * torch.pi)
        bias = bias * (2 * torch.pi)

        x_input = x
        x = torch.addcmul(
            bias.unsqueeze(BATCH_DIM), weight.unsqueeze(BATCH_DIM), x[..., None]
        )
        x = torch.cos(x)
        if self._concat_input:
            # The input goes first so it is always included by the width mask.
            x = torch.cat([x_input[..., None], x], dim=-1)
        x = torch.addcmul(
            elementwise_affine_bias.unsqueeze(BATCH_DIM),
            elementwise_affine_weight.unsqueeze(BATCH_DIM),
            x,
        )

        return x


class _OneHotEncoding(nn.Module):
    """One-hot encoding of categorical features, with support of unknown categories.

    For a categorical feature with cardinality `C` (the number of distinct values), the
    feature values must be from `range(C + 1)`, where the value `C` means "unknown"
    (out-of-vocabulary) category.

    **Shape**

    - Input: `(*, n_categorical_features=len(cardinalities))`.
    - Output: `(*, sum(cardinalities))`.
    """

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0:
            raise ValueError('The categorical input must have at least one dimension')
        if x.shape[-1] != len(self._cardinalities):
            raise ValueError(
                'The input must contain one categorical feature for each entry in'
                ' `cardinalities`'
            )

        return torch.cat(
            [
                # The following produces the standard one-hot encoding for in-vocabulary
                # values, and all-zeros encoding for out-of-vocabulary values. The
                # latter requires the unknown categories to be encoded as `cardinality`.
                nn.functional.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


def _make_num_module(
    type: type[NumericalFeatureEmbeddingsPack], **kwargs
) -> ModulePack:
    cls = type
    del type

    return cls(**kwargs)


class TabPack(ModulePack):
    """A pack of multilayer perceptrons for tabular data, with optional feature embeddings.

    Args:
        n_numerical_features: the number of numerical features.
        categorical_cardinalities: the number of known categories for each categorical
            feature, in input column order. Use an empty list for numerical-only data.
            Encode categories as consecutive integers starting at zero; use the
            cardinality for unknown categories, which receive an all-zero encoding.
        d_out: the number of raw outputs per model.
        n_blocks: the number of MLP blocks, shared or per model.
        d_block: the MLP block width, shared or per model.
        dropout: the dropout rate, shared or per model.
        pack_size: the number of models evaluated in parallel.
        numerical_embeddings: the numerical embedding configuration. The `"type"` entry
            selects an embedding class, such as `CosineEmbeddingsPack`; the remaining
            entries are its keyword arguments, including `d_embedding`. The feature
            count and pack size are supplied automatically. `None` means passing the raw
            numerical features directly to the MLP backbones.
        **backbone_kwargs: optional arguments for `MLPBackbonePack`.
    """  # noqa: E501

    def __init__(
        self,
        *,
        n_numerical_features: int,
        categorical_cardinalities: list[int],
        d_out: int,
        n_blocks: int | list[int],
        d_block: int | list[int],
        dropout: float | list[float],
        pack_size: int,
        numerical_embeddings: KWArgs | None = None,
        **backbone_kwargs,
    ) -> None:
        super().__init__(pack_size=pack_size)

        if numerical_embeddings is None:
            self.num_module = None
            # The same representation size of numerical features for all pack members.
            d_in_num = n_numerical_features
            max_d_in_num = None
        else:
            self.num_module = _make_num_module(
                n_features=n_numerical_features,
                pack_size=pack_size,
                **numerical_embeddings,
            )
            d_num_embedding = numerical_embeddings['d_embedding']
            if isinstance(d_num_embedding, int):
                # The same representation size of numerical features for all members.
                d_in_num = d_num_embedding * n_numerical_features
                max_d_in_num = None
            else:
                # The representation size of numerical features varies between members.
                if not isinstance(d_num_embedding, list):
                    raise TypeError(
                        'The `d_embedding` setting in `numerical_embeddings` must be an'
                        ' integer or a list'
                    )

                max_d_num_embedding = numerical_embeddings.get('max_d_embedding')
                if max_d_num_embedding is None:
                    max_d_num_embedding = max(d_num_embedding)
                d_in_num = [x * n_numerical_features for x in d_num_embedding]
                max_d_in_num = max_d_num_embedding * n_numerical_features

        self.cat_module = (
            _OneHotEncoding(categorical_cardinalities)
            if categorical_cardinalities
            else None
        )
        d_in_cat = sum(categorical_cardinalities)

        self.pack_view = PackView(pack_size=pack_size)
        self.backbone = MLPBackbonePack(
            d_in=(
                d_in_num + d_in_cat
                if isinstance(d_in_num, int)
                else [x + d_in_cat for x in d_in_num]
            ),
            max_d_in=None if max_d_in_num is None else max_d_in_num + d_in_cat,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            pack_size=pack_size,
            **backbone_kwargs,
        )
        self.output = LinearPack(
            d_block,
            d_out,
            max_in_features=backbone_kwargs.get('max_d_block'),
            pack_size=pack_size,
        )

        self._n_numerical_features = n_numerical_features
        self._categorical_cardinalities = categorical_cardinalities

    @property
    def pack_size(self) -> int:
        return self.backbone.pack_size

    def replace(
        self,
        *,
        pack_idx: Tensor,
        numerical_embeddings: KWArgs | None = None,
        d_block: list[int] | None = None,
        **kwargs,
    ) -> None:
        if numerical_embeddings is not None and self.num_module is None:
            raise RuntimeError(
                'Cannot replace numerical embeddings because this model has none'
            )

        if self.num_module is not None:
            self.num_module.replace(
                pack_idx=pack_idx,
                **({} if numerical_embeddings is None else numerical_embeddings),
            )  # ty: ignore[call-non-callable]

        d_num_embedding = (
            None
            if numerical_embeddings is None
            else numerical_embeddings.get('d_embedding')
        )
        self.backbone.replace(
            pack_idx=pack_idx,
            d_in=(
                None
                if d_num_embedding is None
                else [
                    x * self._n_numerical_features
                    + sum(self._categorical_cardinalities)
                    for x in d_num_embedding
                ]
            ),
            d_block=d_block,
            **kwargs,
        )

        self.output.replace(pack_idx=pack_idx, in_features=d_block)

    def forward(
        self, x_num: Tensor | None, x_cat: Tensor | None, pack_idx: Tensor | None = None
    ) -> Tensor:
        """Return raw outputs for each model, with a leading model dimension.

        Pass a shared batch of features, or a separate batch per model during training.
        Use `None` for a feature group that was not configured.
        """
        if x_num is None and x_cat is None:
            raise ValueError('Provide at least one of `x_num` and `x_cat`')
        if pack_idx is not None:
            check_pack_idx(pack_idx, self.pack_size)

        x_list: list[Tensor] = []

        # For models without `num_module`, when the input batch is shared between pack
        # members (e.g., on inference) the input representations passed to the MLP
        # backbones are identical for all pack members, which makes it possible to pass
        # a repeated view of this representation to the backbones instead of a pack of
        # true representation copies. Technically, this is achieved by deferring the
        # use of the `pack_view` module.
        pack_view_used = False

        if x_num is None:
            if self._n_numerical_features != 0:
                raise ValueError(
                    '`x_num` is required when numerical features are configured'
                )
            if self.num_module is not None:
                raise ValueError(
                    '`x_num` is required when numerical embeddings are configured'
                )
        else:
            if self._n_numerical_features <= 0:
                raise ValueError(
                    '`x_num` can only be provided when numerical features are'
                    ' configured'
                )

            if self.num_module is None:
                x_list.append(x_num)
            else:
                x_num_ = self.pack_view(x_num, pack_idx)
                pack_view_used = True
                x_num_ = self.num_module(x_num_, pack_idx)
                x_list.append(x_num_.flatten(-2))
                del x_num_

        if x_cat is None:
            if self.cat_module is not None:
                raise ValueError(
                    '`x_cat` is required when categorical features are configured'
                )
        else:
            if self.cat_module is None:
                raise ValueError(
                    '`x_cat` can only be provided when categorical features are'
                    ' configured'
                )

            x_cat_ = self.cat_module(x_cat).to(torch.get_default_dtype())
            if pack_view_used:
                x_cat_ = self.pack_view(x_cat_, pack_idx)
            x_list.append(x_cat_)
            del x_cat_

        if len(x_list) > 1:
            if (
                x_list[0].device.type == 'mps'
                and _TORCH_IS_OLDER_THAN_2_15
                and sum(x.numel() for x in x_list) > 2**31
            ):
                raise RuntimeError(
                    'MPS concatenation output larger than INT_MAX is not supported'
                    ' with PyTorch < 2.15'
                )
            x = torch.cat(x_list, dim=-1)
        else:
            x = x_list[0]

        if not pack_view_used:
            x = self.pack_view(x, pack_idx)
        x = self.backbone(x, pack_idx)
        x = self.output(x, pack_idx)
        return x
