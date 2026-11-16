# common/: one way to run, time and profile TabPack

Every experiment goes through `python -m common.run CONFIG`, which calls the real
`tabpackage.train`, so there are no copied training loops. Profiling is attached through
small hooks in the library (`src/tabpackage/_instrument.py`). These hooks do nothing
unless profiling is turned on, so results stay comparable across people, kernels and GPUs.

```
common/
  run.py           CLI: train or infer, with optional profiling. Defines VARIANTS (where kernels plug in)
  profiling.py     Instrumenter: timing | torch | nvtx | memory
  data.py          datasets: synthetic, california-sklearn, TabReD (maps-routing, homecredit-default, ...)
  configs/*.json   experiment configs
  tests/           tests (run them in Docker or in the local env)
  docs/heterogeneity-overhead.md   where heterogeneity costs time (forward, backward, optimizer)
```

## 0. Environment

There are two ways to get the same environment: the Docker image (option A) or a local
virtualenv without Docker (option B). Both use the same pinned packages
(`docker/requirements.txt` and `docker/requirements-dev.txt`), so numbers are comparable.

| | A. Docker | B. Local virtualenv |
|---|---|---|
| needs | Docker access (`docker` group), NVIDIA container toolkit | [uv](https://docs.astral.sh/uv/), nothing else (no root, no CUDA toolkit) |
| torch / CUDA | 2.15.0 + CUDA 13.2 (driver ≥ 580) | same by default; older drivers: see options below |
| nsys / ncu | included | **not included**: use system-installed Nsight, or option A |
| ncu on hosts that restrict GPU counters to root | `docker/run.sh --root` | only if you have root (or the admin sets `NVreg_RestrictProfilingToAdminUsers=0`) |

### A. Docker

The single image works on A4000 (sm_86), Hopper (sm_90) and Blackwell (sm_100/sm_120).
It contains CUDA 13.2, torch 2.15.0+cu132, Triton, Nsight Systems and Nsight Compute.
The host needs NVIDIA driver ≥ 580.

The image is published on Docker Hub as `abusagit/tabpack:cu132` (also tagged
`abusagit/tabpack:torch2.15.0-cu132`), so you don't need to build it:

```bash
export IMAGE=abusagit/tabpack:cu132   # docker/run.sh uses $IMAGE (default: locally built tabpack:cu132)
docker pull $IMAGE                    # once: 6.9 GB download, 22 GB on disk
export CUDA_VISIBLE_DEVICES=0         # pick a FREE GPU first (nvidia-smi / gpu_check on shared boxes)
docker/run.sh pytest common/tests -q  # must pass before you trust numbers
docker/run.sh                         # interactive shell
```

To build it yourself instead (after changing `docker/`): `docker/build.sh`, which tags
`tabpack:cu132`; leave `IMAGE` unset to use it.

The repo is mounted at `/workspace`, so edits on the host take effect immediately and you
don't need to rebuild. All outputs go to `.cache/` in the repo, owned by your own user.

### B. Local virtualenv (no Docker)

```bash
env/setup.sh                          # once: creates .venv-tabpack with uv (about 6 GB)
source env/activate.sh                # every new shell; sets PYTHONPATH and caches like the image
export CUDA_VISIBLE_DEVICES=0         # pick a FREE GPU first
pytest common/tests -q                # run from the repo root
python -m common.run common/configs/synthetic_small.json
```

- Run everything **from the repo root**: configs, outputs (`.cache/`) and tests use
  repo-relative paths.
- `env/activate.sh` does what the image does: it puts `src/`, `dev/pepilica/src/` and the
  repo on `PYTHONPATH` (no install needed, edits take effect immediately), and sets
  `TRITON_DEFAULT_FP_FUSION=1`, `TORCH_CUDA_ARCH_LIST` and the Triton/Inductor/ruff caches
  under `.cache/`.
- This is separate from the repo's own `uv sync` environment (`.venv`, `uv.lock`), which
  pins torch 2.7.1+cu126 and can't run on Blackwell.

Options for `env/setup.sh` (environment variables):

| variable | default | meaning |
|---|---|---|
| `TORCH_VERSION` / `TORCH_INDEX` | `2.15.0` / `cu132` | torch build; needs driver ≥ 580. `2.14.1` / `cu130` also needs ≥ 580; `2.11.0` / `cu128` for drivers 570–579 (latest torch built for CUDA 12.8) |
| `PYTHON` | `3.12` | Python version (uv downloads it if missing) |
| `DEV` | `1` | `0` skips the dev tools (onnx, ruff, ty, polars, xdoctest) |
| first argument | `.venv-tabpack` | where to create the venv; pass the same path to `source env/activate.sh PATH` |

Example: `TORCH_VERSION=2.11.0 TORCH_INDEX=cu128 env/setup.sh .venv-cu128`. Results from
a different torch or CUDA build aren't directly comparable: say which one you used.

## 1. Data

| `dataset` in config | where it is read from |
|---|---|
| any, with `"data": "path.npz"` | one `.npz` with `<split>.x_num`, `<split>.x_cat`, `<split>.y` arrays (e.g. written by `dev/nosorozhek/scripts/prepare_tabred.py`); preprocessed like raw TabReD |
| `synthetic` | generated on the fly; size is set in the config's `synthetic` block |
| `california-sklearn` | downloaded automatically by scikit-learn into `data/california-sklearn/` on first use |
| TabReD: `maps-routing`, `homecredit-default`, `cooking-time`, `delivery-eta`, `ecom-offers`, `homesite-insurance`, `sberbank-housing`, `weather` | `data/tabred/<name>/` as downloaded below: `x_num.npy`, `x_bin.npy`, `x_cat.npy`, `y.npy`, `info.json`, `splits/default/{train,val,test}.npy`. For `maps-routing` and `homecredit-default`, a paper-recipe cache in `.cache/maps-routing-paper/` or `.cache/homecredit-paper/` (from `dev/pepilica/scripts/preprocess_maps_paper.py`) takes priority |

Data is looked up in `data/` at the repo root (gitignored), then `.cache/data/`; a config can set `"data_root"` to use another directory.

**Preprocessed cache.** Preprocessing a TabReD dataset takes up to ~80 s (Maps Routing).
The first run caches the result in `data/preprocessed/<name>-seed<seed>.npz` (plus a
`.json` with metadata), and later runs load it in seconds. It's rebuilt automatically when
any source file changes (re-download) or the preprocessing code changes
(`PREPROCESS_VERSION` in `common/data.py`). Delete `data/preprocessed/` to force a
rebuild, or set `"cache_preprocessed": false` in a config to bypass it. Synthetic and
California data aren't cached; they take about a second.

For raw directories, numeric features are quantile-normalised, binary and categorical features are ordinal-encoded, and the regression target is standardised (train split only).

### Downloading TabReD datasets

Datasets are **not downloaded during a run**. Download them once, before the first run, with
Yandex's official [`tabred`](https://github.com/yandex-research/tabred) tool. Run it on the
host, from the repo root (it needs [uv](https://docs.astral.sh/uv/)), so the files land in
`data/tabred/` (gitignored), where both the Docker container and the local env find them:

```bash
uvx --with six tabred download maps-routing --output-path data/tabred      # one dataset
uvx --with six tabred download maps-routing homecredit-default --output-path data/tabred
uvx --with six tabred download all --output-path data/tabred               # all eight
```

- `--with six` works around a packaging bug in `kaggle` 2.2.5: it imports `six` without
  declaring it, so a plain `uvx tabred ...` fails with `No module named 'six'`.
- It needs a **Kaggle API token** in `~/.kaggle/kaggle.json`: Kaggle → Settings → API → *Create New Token*.
- `homecredit-default`, `homesite-insurance`, `sberbank-housing` and `ecom-offers` come
  from Kaggle competitions. Your Kaggle account must first accept each competition's rules
  on its Kaggle page, otherwise the download is refused. The tool tells you which ones.
- If a dataset is missing, `common.run` stops with the exact list of paths it checked and
  this command.

## 2. Run

```bash
# torch.profiler trace only (no nsys needed): writes trace.json, operators.txt, trace-summary.json
docker/run.sh python -m common.run common/configs/maps_routing.json --profile torch

# training; by default it times each phase in 5 steps of epoch `profile_epoch`, then stops
docker/run.sh python -m common.run common/configs/maps_routing.json

# choose the window: epoch 3, 20 steps; --steps 0 covers the whole epoch including eval
docker/run.sh python -m common.run common/configs/maps_routing.json --epoch 3 --steps 20

# torch profiler schedule: from epoch 3, 3 cycles of (2 wait, 1 warmup, 5 recorded) steps
# -> trace-0.json, trace-1.json, trace-2.json (one per cycle)
docker/run.sh python -m common.run common/configs/maps_routing.json --profile torch \
    --epoch 3 --wait 2 --warmup 1 --steps 5 --repeat 3

# inference: train 1 epoch (--max-epochs), then profile the predictor on test-split batches
docker/run.sh python -m common.run common/configs/maps_routing.json --mode infer --profile torch

# quick run that doesn't need any data
docker/run.sh python -m common.run common/configs/synthetic_small.json
```

### Schedule: which steps are recorded

The same options work for every `--profile` mode and mean the same as
[`torch.profiler.schedule`](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule).
Starting at the first step of `--epoch`, a cycle of `wait + warmup + steps` steps is run
`--repeat` times, and only the `--steps` steps of each cycle are recorded:

```
--epoch E   --wait M --warmup Y --steps N --repeat X
            |  M skipped  |  Y run, not recorded  |  N recorded  |  x X cycles
```

| option | default | meaning |
|---|---|---|
| `--epoch E` | config `profile_epoch` (or 3) | the schedule starts at the first step of epoch E (train mode). In infer mode it starts at the first test batch, and its wait/warmup steps are the warmup |
| `--wait M` | 0 | steps skipped at the start of each cycle |
| `--warmup Y` | 1 (0 with `--steps 0`) | steps run but not recorded. Profilers skew the first step after they start, so keep at least 1 |
| `--steps N` | 5 | recorded steps per cycle. `0` = the rest of the epoch, including `eval` and the epoch glue (not combinable with wait/warmup/repeat) |
| `--repeat X` | 1 | number of cycles |

What each mode records over the schedule:
- `timing` and `memory`: only the recorded steps. With `--repeat > 1`, `report.json` also
  has `phase_ms_per_cycle`.
- `torch`: one trace per cycle: `trace.json`, `operators.txt`, `trace-summary.json`, or
  `trace-<k>.json` etc. when `--repeat > 1`.
- `nvtx`: one capture range (`cudaProfilerStart/Stop`) per cycle; see Nsight Systems below.

The schedule can run past the end of an epoch into the next one.

**Run length.** A profiled training run (any `--profile` except `none`) stops at the end
of the epoch in which the schedule finishes, so you don't wait for full training. Use
`--full-training` to train for the config's `max_epochs` anyway, or `--max-epochs K` to
set it explicitly. `--profile none` always uses the config's `max_epochs`. Infer mode
trains 1 epoch (`--max-epochs`), then runs exactly the schedule's steps over the test
batches (one pass with `--steps 0`). `report.json` records the `max_epochs` used.

`torch.profiler` options (`--profile torch`):

| option | default | meaning |
|---|---|---|
| `--torch-record-shapes` / `--no-torch-record-shapes` | on | record operator input shapes (`operators.txt` groups by shape) |
| `--torch-profile-memory` / `--no-torch-profile-memory` | on | record tensor allocations (memory timeline in the trace) |
| `--torch-with-stack` / `--no-torch-with-stack` | on | record Python stacks for every op (source lines in the trace; much larger traces and more profiler overhead, so turn it off for clean timings) |

### `--profile` modes

| mode | what it does | output in `--out` (default `.cache/runs/<name>-<mode>-<profile>-<variant>-<time>/`) |
|---|---|---|
| `none` | plain run, no hooks | `report.json` (wall time, peak memory) |
| `timing` (default) | CUDA events around each phase | `report.json → train_profile.phase_ms` (median, mean and total ms per phase) |
| `torch` | `torch.profiler` with `tabpack/<phase>` ranges | `trace.json` (open in https://perfetto.dev or `chrome://tracing`), `operators.txt`, `trace-summary.json` (kernels per phase) |
| `memory` | peak allocated memory per phase (synchronises, so don't use it for timing) | `report.json → phase_memory_mib` |
| `nvtx` | NVTX `tabpack.<phase>` ranges plus `cudaProfilerStart/Stop` around the window (or, with `--capture-phase P`, around phase P only); use it under nsys or ncu (see below) | whatever nsys or ncu writes |

### Phases

Every measurement is split into the phases below. The names are the same in every
mode: `phase_ms` and `phase_memory_mib` keys in `report.json`, `tabpack/<phase>` ranges
in torch traces, and `tabpack.<phase>` NVTX ranges in nsys and ncu. The hooks are in
`src/tabpackage/_training.py` (`_train`, search for `_instrument.`).

| phase | when | what runs inside | heterogeneity overhead (see `docs/heterogeneity-overhead.md`) |
|---|---|---|---|
| `gather` | every step | `batch = {k: v[batch_idx]}`: each member gets its own shuffled rows, a `(P, B)` index gather copying P×B rows | constant cost, large on wide tables |
| `forward` | every step | `TabPack` forward under BF16 autocast: embeddings, padded `bmm`/`baddbmm` per block, output masks, per-member dropout, depth handling, head | padded GEMMs (all members cost as much as the widest), dropout masks, depth `where`/index copies |
| `loss` | every step | `loss_fn(y_pred, y)`, one loss per member | negligible |
| `backward` | every step | `losses.sum().backward()`: dX and dW of every padded GEMM, mask backward | about 2× the forward padding waste. Its kernels run on autograd's worker thread; see the ncu notes |
| `optimizer` | every step | `MuonAdamWPack.step()`: Muon (momentum + 5 Newton–Schulz iterations per block) for hidden weights, a per-parameter AdamW loop for the rest | **usually the largest phase** (60–75%): Newton–Schulz on padded matrices, AdamW launch overhead |
| `zero_grad` | every step | `optimizer.zero_grad()` (sets grads to `None`) | ~0 |
| `eval` | once per epoch, after the last step | evaluation of all members on `val` (padded forward in eval mode) | same padding as forward. Only inside the window with `--steps 0` |
| `infer` | per batch, `--mode infer` only | `TabPackPredictor` forward of the trained ensemble on the `test` split | inference cost of the selected ensemble members |

Not a phase: the epoch glue after `eval` (checkpoint copies, greedy ensemble updates,
replacing or removing finished members). With `--steps 0` it shows up as the gap between
`eval` and the next `gather` (`window_ms` minus the sum of the phases in timing mode).

`timing` uses CUDA events, so each phase's time is GPU time between its start and end
markers, including idle gaps if the CPU can't launch kernels fast enough. For launch-bound
phases (AdamW, small packs) also look at the torch trace, which shows CPU-side gaps.

### Nsight Systems (timeline)

```bash
docker/run.sh nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o .cache/runs/maps-nsys python -m common.run common/configs/maps_routing.json --profile nvtx
```
Open `.cache/runs/maps-nsys.nsys-rep` in the Nsight Systems GUI. For a text summary run
`nsys stats .cache/runs/maps-nsys.nsys-rep`.

The capture follows the schedule: there is one capture range per cycle. To get one report
per cycle, use `--capture-range-end=repeat:X` with the same X as `--repeat`. This writes
`maps-nsys.1.nsys-rep` … `maps-nsys.X.nsys-rep`:

```bash
docker/run.sh nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=repeat:3 -o .cache/runs/maps-nsys \
    python -m common.run common/configs/maps_routing.json --profile nvtx \
    --wait 2 --warmup 1 --steps 5 --repeat 3
```

All other `nsys profile` options (`--trace`, `--sample`, `--gpu-metrics-devices`,
`--cuda-memory-usage`, …) go before `python` as usual, since nsys is just the wrapping
command. The same holds for ncu below.

### Nsight Compute (per-kernel)

ncu needs GPU counter access. On our hosts only root has it, so use `--root` or contact your system administrator to enable GPU counters. Files are still handed back to your user afterwards.

Profile one phase of one step. The recommended way works for every phase, including
`backward`:

```bash
docker/run.sh --root ncu --profile-from-start off --set full \
    -o .cache/runs/maps-backward \
    python -m common.run common/configs/maps_routing.json --profile nvtx --steps 1 \
        --capture-phase backward
```
- `--capture-phase P` opens the profiler capture range (`cudaProfilerStart/Stop`) around
  phase `P` only. That range is process-wide, so it also catches kernels launched from
  autograd's worker thread.
- Don't use `--nvtx-include "tabpack.backward/"`. NVTX ranges belong to the thread that
  opened them, and backward kernels are launched from autograd's worker thread, so the
  filter catches only 2 of the ~47 backward kernels. For phases that run on the main
  thread (`forward`, `optimizer`, `gather`, ...) an NVTX filter does work:
  `ncu --nvtx --nvtx-include "tabpack.optimizer/" --profile-from-start off ...`. The
  trailing `/` marks a push/pop range. NVTX names use `.` because ncu reads `/` as
  nesting.
- Add `-k regex:<kernel name>` to target one kernel and `--launch-count N` to cap the
  number of kernels. Keep `--steps` small, because ncu replays every kernel many times.
- Open the `.ncu-rep` in the Nsight Compute GUI, or run `ncu -i FILE.ncu-rep --page details`.

## 3. Configs

A config is a JSON file with these fields:
- `dataset` and `seed`;
- `n_models`, plus the per-model search space: `widths`, `depth_min`/`depth_max`,
  `dropout_max`, log-uniform ranges for `adamw_lr`, `muon_lr` and `weight_decay`, and
  optionally `embedding_dims` + `init_scale`, which turn on cosine numerical embeddings;
- `batch_size`, `max_epochs` (`null` means run until the ensemble finalises),
  `model_patience`, `ensemble_patience`, `max_ensemble_size`;
- `inference_batch_size` and `profile_epoch`;
- optional: `forward_impl` (`indexed` by default, or `dense`), `muon_skip_zero_updates`
  (Ivan's flag), `autocast` (`"bfloat16"` or `null`), `data_root`, `cache_preprocessed`
  (default `true`).

Copy an existing config and change it; don't add command-line flags.

| config | dataset | what | from |
|---|---|---|---|
| `synthetic_small` | synthetic | tiny, about 10 s | common |
| `synthetic_hetero_1024` | synthetic | 64 members, widths 64–1024 | common |
| `california` | `california-sklearn` | 64 members, widths 32–512, embeddings | `dev/pepilica` (same hyperparameters as PR #2's `diverse_california`) |
| `california_diverse_768` | `california-sklearn` | 64 members, widths 32–768, embeddings | PR #1 |
| `california_diverse_compile` | `california-sklearn` | `california` hyperparameters, `compile_model: true` | PR #2 |
| `california_paper_uniform384` | `california-sklearn` | paper setup: uniform width 384, depth 1–3 | PR #2 |
| `california_paper_uniform384_compile` | `california-sklearn` | the same, `compile_model: true` | PR #2 |
| `maps_routing`, `maps_routing_1024` | `maps-routing` | 64 members, widths 64–512 / 64–1024 | `dev/pepilica` |
| `maps_routing_small` | `maps-routing` | quick: 8 members, widths 32–128, 3 epochs | PR #1 |
| `homecredit` | `homecredit-default` | `maps_routing` setup on Homecredit | `dev/pepilica` |

All California configs use `california-sklearn`: California Housing (20640 rows, 8
features) from scikit-learn, downloaded automatically, with PR #2's seeded random
64/16/20 train/val/test split. (`dev/pepilica` used the tabpack-data archive's fixed
split instead: same data, different rows per split.)

Other supported keys:
- `autocast_dtype`: same as `autocast` (used by the PR #1 configs);
- `compile_model`, `compile_model_numerical_embeddings`, `compile_model_first_block`,
  `compile_optimizer_muon_newton_schulz` (the `*_compile` configs set `compile_model`);
- `data` + `task_type`: read one `.npz` with `<split>.x_num`, `<split>.x_cat`,
  `<split>.y` arrays instead of `dataset` (e.g. from
  `dev/nosorozhek/scripts/prepare_tabred.py`). If the file doesn't exist, `dataset` is
  used.

## 4. Adding a variant (your kernel)

`common/run.py` has one registry, `VARIANTS: name -> fn(train_kwargs) -> train_kwargs`.
To compare your implementation against `native`:

```python
def _my_kernel(kwargs):
    import my_kernels  # e.g. from dev/<you>/src, already on PYTHONPATH as a package

    my_kernels.install()  # e.g. swap LinearPack.forward / the optimizer implementation
    return kwargs  # or change kwargs, e.g. {**kwargs, 'optimizer': {...}}


VARIANTS = {'native': _native, 'compiled': _compiled, 'my_kernel': _my_kernel}
```
Then run `python -m common.run CONFIG --variant my_kernel`, using the same config, window
and GPU as the baseline. The phase names are unchanged, so `report.json` files and traces
can be compared directly.

## 5. Comparing runs

Every `report.json` records the environment (GPU, compute capability, torch, CUDA,
Triton), the config, the per-phase numbers, the peak memory and the wall time. When you
post results, include the run directory name and the GPU it ran on.
