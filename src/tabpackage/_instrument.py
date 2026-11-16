"""Optional instrumentation hooks for the training loop (profiling, timing, NVTX).

By default no instrumenter is installed and every hook is a no-op, so training
behaviour and performance are unchanged. Tools such as `common/profiling.py` install an
object implementing the `Instrumenter` protocol via `set_instrumenter`.
"""

import contextlib
from typing import Any, Protocol


class Instrumenter(Protocol):
    def phase(self, name: str) -> contextlib.AbstractContextManager[Any]:
        """Wrap one named phase of the training loop (e.g. ``'forward'``)."""
        ...

    def step_end(self) -> None:
        """Called after every training step (after ``optimizer.zero_grad``)."""
        ...

    def epoch_end(self) -> None:
        """Called at the end of every epoch (after evaluation and ensembling)."""
        ...


_INSTRUMENTER: Instrumenter | None = None
_NULL = contextlib.nullcontext()


def set_instrumenter(instrumenter: Instrumenter | None) -> None:
    """Install (or, with ``None``, remove) the global instrumenter."""
    global _INSTRUMENTER
    _INSTRUMENTER = instrumenter


def get_instrumenter() -> Instrumenter | None:
    return _INSTRUMENTER


def phase(name: str) -> contextlib.AbstractContextManager[Any]:
    return _NULL if _INSTRUMENTER is None else _INSTRUMENTER.phase(name)


def step_end() -> None:
    if _INSTRUMENTER is not None:
        _INSTRUMENTER.step_end()


def epoch_end() -> None:
    if _INSTRUMENTER is not None:
        _INSTRUMENTER.epoch_end()
