"""PyTorch utilities."""

import dataclasses
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import Tensor

_T = TypeVar('_T')
_R = TypeVar('_R')
_IDX = TypeVar('_IDX', bound=int | slice | list | Tensor)


# >>> Helpers


def _map(fn: Callable[[Tensor], Tensor], data: _T) -> _T:
    if isinstance(data, Tensor):
        return fn(data)  # ty:ignore[invalid-return-type]

    elif data is None:
        return data

    elif isinstance(data, tuple | list):
        constructor = type(data)
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        return constructor(_map(fn, x) for x in data)

    elif isinstance(data, set | frozenset):
        return type(data)(_map(fn, x) for x in data)

    elif isinstance(data, dict):
        return type(data)((k, _map(fn, v)) for k, v in data.items())

    elif dataclasses.is_dataclass(data):
        return type(data)(
            **{
                field.name: _map(fn, getattr(data, field.name))
                for field in dataclasses.fields(data)
            }
        )

    else:
        raise TypeError(f'The input type {type(data)} is not supported.')


def _apply2(fn: Callable[[Tensor, Tensor], _R], a: _T, b: _T) -> None:
    if type(a) is not type(b):
        raise TypeError(
            'The values and destinations have incompatible nested structures'
        )

    if isinstance(a, Tensor):
        fn(a, b)  # ty:ignore[invalid-argument-type]

    elif isinstance(a, tuple | list):
        for x, y in zip(a, b, strict=True):  # ty:ignore[invalid-argument-type]
            _apply2(fn, x, y)

    elif isinstance(a, dict):
        if a.keys() != b.keys():  # ty:ignore[unresolved-attribute]
            raise ValueError(
                'The values and destinations have different dictionary keys'
            )
        for key in a:
            _apply2(fn, a[key], b[key])  # ty:ignore[not-subscriptable]

    elif dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            _apply2(fn, getattr(a, field.name), getattr(b, field.name))

    else:
        raise TypeError(f'The input type {type(a)} is not supported.')


# >>> Functions


_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
if hasattr(torch, 'uint16'):  # True starting from PyTorch 2.3.
    _INTEGER_DTYPES += (torch.uint16, torch.uint32, torch.uint64)


def is_integer(x: Tensor) -> bool:
    """Check whether a tensor has an integer data type."""
    return x.dtype in _INTEGER_DTYPES


def to(data: _T, *args, **kwargs) -> _T:
    """Change the data type and/or device of all tensors in a collection."""
    return _map(lambda x: x.to(*args, **kwargs), data)


# Reference: https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/_tensor_ops.py#L147
def cat(data: list[_T], dim: int = 0) -> _T:
    """Concatenate matching tensors in nested tuples, dictionaries, or dataclasses."""
    if not data:
        raise ValueError('The input must be non-empty')

    first = data[0]

    if isinstance(first, Tensor):
        return torch.cat(data, dim=dim)  # ty: ignore[no-matching-overload, invalid-argument-type, invalid-return-type, unused-ignore-comment, unused-ignore-comment]

    elif isinstance(first, tuple):
        constructor = type(first)
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        return constructor(
            cat([x[i] for x in data], dim=dim)  # ty: ignore[not-subscriptable, unused-ignore-comment, unused-ignore-comment]
            for i in range(len(first))
        )

    elif isinstance(first, dict):
        return type(first)((key, cat([x[key] for x in data], dim=dim)) for key in first)  # ty: ignore[not-subscriptable]

    elif dataclasses.is_dataclass(first):
        return type(first)(
            **{
                field.name: cat([getattr(x, field.name) for x in data], dim=dim)
                for field in dataclasses.fields(first)
            }
        )

    else:
        raise ValueError(f'The collection type {type(first)} is not supported.')


def index(data: _T, idx: _IDX) -> _T:
    """Index all tensors in an arbitrarily nested collection with the same index."""
    if isinstance(idx, int | slice | list):
        return _map(lambda x: x[idx], data)

    elif isinstance(idx, Tensor):
        device_to_idx = {idx.device: idx}

        def f(x: Tensor) -> Tensor:
            idx_on_device = device_to_idx.get(x.device)
            if idx_on_device is None:
                idx_on_device = idx.to(x.device)
                device_to_idx[x.device] = idx_on_device  # ty:ignore[invalid-assignment]
            return x[idx_on_device]

        return _map(f, data)

    else:
        raise TypeError(f'The index type {type(idx)} is not supported')


def index_set_(data: _T, indices: tuple[Tensor, ...], values: _T) -> None:
    """Put values into all tensors of an arbitrarily nested collection in place."""
    if not indices:
        raise ValueError('`indices` must not be empty')

    device_to_indices = {}

    def f(x: Tensor, value: Tensor) -> None:
        indices_on_device = device_to_indices.get(x.device)
        if indices_on_device is None:
            indices_on_device = tuple(
                idx if idx.device == x.device else idx.to(x.device) for idx in indices
            )
            device_to_indices[x.device] = indices_on_device
        x.index_put_(indices_on_device, value)

    _apply2(f, data, values)
