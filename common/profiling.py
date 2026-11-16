"""One instrumenter for timing / torch.profiler / NVTX (nsys, ncu) / memory.

It is installed into the training loop via ``tabpackage._instrument.set_instrumenter``
and receives these phases every step::

    gather -> forward -> loss -> backward -> optimizer -> zero_grad -> step_end()

plus ``eval`` and ``epoch_end()`` once per epoch. ``common.run`` also uses it for
inference (phase ``infer``).

What is measured is set by a `Schedule`, with the same meaning as
``torch.profiler.schedule``: starting at ``start_step``, each cycle is ``wait`` skipped
steps, ``warmup`` steps (run, not recorded) and ``active`` recorded steps; the cycle is
repeated ``repeat`` times. ``active=0`` means "the rest of the epoch" (one cycle, no
wait/warmup): the window then also covers ``eval`` and the epoch glue (checkpointing,
ensembles, pack replacement).

Per mode, "recorded" means:

* ``timing``  CUDA events around every phase of every active step;
* ``memory``  peak allocated memory per phase of every active step;
* ``torch``   ``torch.profiler.schedule(wait, warmup, active, repeat)``: one trace per
  active block (``trace.json``, or ``trace-<k>.json`` when ``repeat > 1``);
* ``nvtx``    one ``cudaProfilerStart/Stop`` capture range per active block (nsys with
  ``--capture-range-end=repeat:<repeat>`` writes one report per block), and NVTX
  ``tabpack.<phase>`` ranges.
"""

import contextlib
import dataclasses
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

MODES = ('none', 'timing', 'torch', 'nvtx', 'memory')
STEP_START_PHASES = ('gather', 'infer')  # first phase of a training / inference step
PHASES = (
    'gather',
    'forward',
    'loss',
    'backward',
    'optimizer',
    'zero_grad',
    'eval',
    'infer',
)


@dataclasses.dataclass
class Schedule:
    start_step: int
    active: int  # 0 = until the end of the epoch
    wait: int = 0
    warmup: int = 0
    repeat: int = 1

    def __post_init__(self) -> None:
        if min(self.active, self.wait, self.warmup) < 0 or self.repeat < 1:
            raise ValueError(f'invalid schedule: {self}')
        if self.active == 0 and (self.wait or self.warmup or self.repeat != 1):
            raise ValueError(
                'active=0 (whole epoch) cannot be combined with wait, warmup or repeat'
            )

    @property
    def cycle(self) -> int:
        return self.wait + self.warmup + self.active

    @property
    def total(self) -> int:
        """Steps covered by the whole schedule (0 = until the end of the epoch)."""
        return self.cycle * self.repeat

    def where(self, step: int) -> tuple[int, str]:
        """(cycle index, 'wait' | 'warmup' | 'active') of an absolute step."""
        if self.active == 0:
            return 0, 'active'
        k, pos = divmod(step - self.start_step, self.cycle)
        if pos < self.wait:
            return k, 'wait'
        if pos < self.wait + self.warmup:
            return k, 'warmup'
        return k, 'active'


class Instrumenter:
    def __init__(
        self,
        mode: str,
        out: Path,
        schedule: Schedule,
        capture_phase: str | None = None,
        torch_options: dict | None = None,
    ):
        """capture_phase (nvtx mode only): instead of one capture range per active
        block, open cudaProfilerStart/Stop around each occurrence of this phase in
        active steps. Use it for ncu on `backward`: autograd launches its kernels from
        a worker thread, which per-thread NVTX filters (`--nvtx-include`) cannot see,
        while the capture range is process-wide.

        torch_options: `torch.profiler.profile` arguments; record_shapes,
        profile_memory and with_stack default to True."""
        assert mode in MODES, mode
        assert capture_phase is None or mode == 'nvtx', 'capture_phase needs nvtx'
        self.mode, self.out = mode, Path(out)
        self.schedule = schedule
        self.capture_phase = capture_phase
        self.torch_options = {
            'record_shapes': True,
            'profile_memory': True,
            'with_stack': True,
            **(torch_options or {}),
        }
        self.step = 0
        self.active = False  # inside the schedule window
        self.done = False
        self._capturing = False  # nvtx: inside a cudaProfilerStart/Stop range
        self._events = defaultdict(list)  # phase -> [(cycle, start_ev, end_ev)]
        self._memory = defaultdict(list)  # phase -> [{cycle, before, peak, after}]
        self._prof = None
        self._n_traces = 0
        self._window_events = None

    @property
    def start_step(self) -> int:
        return self.schedule.start_step

    # >>> tabpackage._instrument protocol

    def phase(self, name: str):
        # The window opens at the first phase of step `start_step` (not at the previous
        # epoch's `eval`, which runs while the step counter already equals start_step).
        if (
            not self.active
            and not self.done
            and self.step == self.start_step
            and name in STEP_START_PHASES
        ):
            self._begin()
        if not self.active:
            return contextlib.nullcontext()
        if name in STEP_START_PHASES:
            self._maybe_start_capture()
        return self._phase(name)

    def step_end(self) -> None:
        if self.active and self.mode == 'torch' and self.schedule.active:
            self._prof.step()
        if self._capturing and self.capture_phase is None:
            k, kind = self.schedule.where(self.step)
            nxt_k, nxt_kind = self.schedule.where(self.step + 1)
            if kind == 'active' and (nxt_kind != 'active' or nxt_k != k):
                self._stop_capture()
        self.step += 1
        if (
            self.active
            and self.schedule.active
            and self.step >= self.start_step + self.schedule.total
        ):
            self._end()

    def epoch_end(self) -> None:
        if self.active and self.schedule.active == 0:
            self._end()

    # >>> internals

    def _recording(self, name: str) -> tuple[bool, int]:
        """Whether the current step is an active (recorded) one, and its cycle."""
        k, kind = self.schedule.where(self.step)
        if name == 'eval' and self.schedule.active:
            return False, k  # eval is only part of whole-epoch windows
        return kind == 'active', k

    @contextlib.contextmanager
    def _phase(self, name: str):
        recording, k = self._recording(name)
        if self.mode == 'timing' and recording:
            s, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            s.record()
            yield
            e.record()
            self._events[name].append((k, s, e))
        elif self.mode == 'torch':
            # Annotate every step; the profiler schedule decides what is recorded.
            with torch.profiler.record_function(f'tabpack/{name}'):
                yield
        elif self.mode == 'nvtx':
            capture = recording and name == self.capture_phase
            if capture:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
            # '.' not '/': ncu's --nvtx-include treats '/' as a range-nesting separator.
            with torch.cuda.nvtx.range(f'tabpack.{name}'):
                yield
            if capture:
                torch.cuda.synchronize()  # wait for kernels from all threads
                torch.cuda.cudart().cudaProfilerStop()
        elif self.mode == 'memory' and recording:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            yield
            torch.cuda.synchronize()
            self._memory[name].append(
                {
                    'cycle': k,
                    'before': before,
                    'peak': torch.cuda.max_memory_allocated(),
                    'after': torch.cuda.memory_allocated(),
                }
            )
        else:
            yield

    def _maybe_start_capture(self) -> None:
        """nvtx: open a capture range at the first step of each active block."""
        if self.mode != 'nvtx' or self.capture_phase is not None or self._capturing:
            return
        if self.schedule.where(self.step)[1] == 'active':
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            self._capturing = True

    def _stop_capture(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        self._capturing = False

    def _begin(self) -> None:
        self.active = True
        torch.cuda.synchronize()
        if self.mode == 'torch':
            sch = self.schedule
            self._prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=(
                    torch.profiler.schedule(
                        wait=sch.wait,
                        warmup=sch.warmup,
                        active=sch.active,
                        repeat=sch.repeat,
                    )
                    if sch.active
                    else None
                ),
                on_trace_ready=self._save_trace,
                **self.torch_options,
            )
            self._prof.start()
        elif self.mode == 'timing':
            self._window_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self._window_events[0].record()
        print(
            f'[profiling] {self.mode}: window started at step {self.step} '
            f'({self.schedule})',
            flush=True,
        )

    def _end(self) -> None:
        torch.cuda.synchronize()
        self.active, self.done = False, True
        self.out.mkdir(parents=True, exist_ok=True)
        if self.mode == 'torch':
            self._prof.stop()  # whole-epoch window: triggers _save_trace
            self._prof = None
        elif self.mode == 'nvtx' and self._capturing:
            self._stop_capture()
        elif self.mode == 'timing':
            self._window_events[1].record()
            torch.cuda.synchronize()
        print(f'[profiling] {self.mode}: window ended at step {self.step}', flush=True)

    def _save_trace(self, prof) -> None:
        """torch.profiler on_trace_ready: one trace per active block."""
        self.out.mkdir(parents=True, exist_ok=True)
        k = self._n_traces
        self._n_traces += 1
        suffix = '' if self.schedule.repeat == 1 else f'-{k}'
        trace = self.out / f'trace{suffix}.json'
        prof.export_chrome_trace(str(trace))
        table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=80)
        (self.out / f'operators{suffix}.txt').write_text(table + '\n')
        steps = self.schedule.active or max(self.step + 1 - self.start_step, 1)
        _try_summarize(trace, self.out / f'trace-summary{suffix}.json', steps)
        print(f'[profiling] torch: wrote {trace}', flush=True)

    def summary(self) -> dict:
        """Per-phase statistics for report.json (timing / memory modes)."""
        res: dict = {
            'mode': self.mode,
            'schedule': dataclasses.asdict(self.schedule),
        }
        if self.mode == 'torch':
            res['traces'] = self._n_traces
        if self.mode == 'timing' and self._events:
            torch.cuda.synchronize()
            ms = {
                name: [(k, s.elapsed_time(e)) for k, s, e in evs]
                for name, evs in self._events.items()
            }
            res['phase_ms'] = {
                name: _stats([v for _, v in x]) for name, x in ms.items()
            }
            if self.schedule.repeat > 1:
                res['phase_ms_per_cycle'] = [
                    {
                        name: _stats(vals)
                        for name, x in ms.items()
                        if (vals := [v for c, v in x if c == k])
                    }
                    for k in range(self.schedule.repeat)
                ]
            if self._window_events and self.done:
                res['window_ms'] = self._window_events[0].elapsed_time(
                    self._window_events[1]
                )
        if self.mode == 'memory' and self._memory:
            mib = 2**20
            res['phase_memory_mib'] = {
                name: {
                    'max_peak': max(r['peak'] for r in rs) / mib,
                    'max_peak_over_before': max(r['peak'] - r['before'] for r in rs)
                    / mib,
                    'mean_after_minus_before': statistics.fmean(
                        r['after'] - r['before'] for r in rs
                    )
                    / mib,
                }
                for name, rs in self._memory.items()
            }
        return res


def _stats(ms: list[float]) -> dict:
    return {
        'median': statistics.median(ms),
        'mean': statistics.fmean(ms),
        'total': sum(ms),
        'count': len(ms),
    }


def _try_summarize(trace: Path, out: Path, steps: int) -> None:
    """Per-phase kernel summary via the existing dev/pepilica helper (if importable)."""
    try:
        from tabpack_experimental.benchmark import summarize_trace
    except ImportError:
        return
    try:
        summary = summarize_trace(json.loads(trace.read_text()), steps, True, PHASES)
        out.write_text(json.dumps(summary, indent=2, default=str) + '\n')
    except Exception as e:  # noqa: BLE001  (a convenience; never fail the run on it)
        print(f'[profiling] trace summary skipped: {e!r}', flush=True)
