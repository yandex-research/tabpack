"""Unified TabPack launcher: train or infer, optionally profiled.

    python -m common.run CONFIG.json [--mode train|infer] [--profile MODE]
                         [--epoch E] [--wait M] [--warmup Y] [--steps N] [--repeat X]
                         [--out DIR] [--max-epochs K] [--variant native|compiled]

See common/README.md for the full instructions, nsys/ncu wrappers and outputs.
"""

import argparse
import datetime
import functools
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch

import tabpackage
import tabpackage._instrument
import tabpackage.ensemble
import tabpackage.nn
import tabpackage.optim
from common import data
from common.profiling import MODES, Instrumenter, Schedule

# >>> Variants: the single hook point for alternative implementations / kernels.
# A variant receives the `tabpackage.train` kwargs and returns them (possibly modified).
# To plug in a kernel, add an entry here (see README, "Adding a variant").


def _native(kwargs: dict) -> dict:
    return kwargs


def _compiled(kwargs: dict) -> dict:
    return {
        **kwargs,
        'compile_model': True,
        'compile_optimizer_muon_newton_schulz': True,
    }


VARIANTS = {'native': _native, 'compiled': _compiled}


# >>> Config -> tabpackage.train arguments


def sample_hyperparameters(cfg: dict) -> list[dict]:
    """Per-model hyperparameters (same sampling as dev/pepilica/scripts/run_*.py)."""
    rng = random.Random(cfg['seed'])

    def loguniform(bounds):
        return math.exp(rng.uniform(*map(math.log, bounds)))

    result = []
    for _ in range(cfg['n_models']):
        model = {
            'd_block': rng.choice(cfg['widths']),
            'n_blocks': rng.randint(cfg['depth_min'], cfg['depth_max']),
        }
        if cfg.get('embedding_dims'):
            model['numerical_embeddings'] = {
                'd_embedding': rng.choice(cfg['embedding_dims']),
                'init_scale': loguniform(cfg['init_scale']),
            }
        model['dropout'] = (
            rng.uniform(0, cfg['dropout_max']) if rng.randint(0, 1) else 0.0
        )
        optimizer = {
            k: loguniform(cfg[k]) for k in ('adamw_lr', 'muon_lr', 'weight_decay')
        }
        result.append({'model': model, 'optimizer': optimizer})
    return result


def make_task(meta: dict):
    """Loss, eval metric and prediction transform for the dataset's task."""
    if meta['task'] == 'binclass':

        def loss_fn(y_pred, y):  # (P, B, 1), (P, B) -> (P,)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                y_pred.squeeze(-1), y.float(), reduction='none'
            ).mean(1)

        def metric(*, y_true, y_pred):
            # Log-loss over the last dim: y_pred is (P, N) for models and (N,) when
            # scoring an ensemble, which must give a scalar.
            p = y_pred.float().clamp(1e-7, 1 - 1e-7)
            y = y_true.to(p.dtype)
            return -(y * p.log() + (1 - y) * (1 - p).log()).mean(-1)

        transform = functools.partial(_sigmoid_squeeze)
        return loss_fn, metric, transform, 'logloss'

    scale = meta['target_std']

    def loss_fn(y_pred, y):
        return (y_pred.squeeze(-1) - y).square().mean(1)

    def metric(*, y_true, y_pred):
        return (y_pred - y_true).square().mean(-1).sqrt() * scale

    return loss_fn, metric, functools.partial(torch.squeeze, dim=-1), 'rmse'


def _sigmoid_squeeze(x):
    return torch.sigmoid(x.squeeze(-1))


def build_train_kwargs(cfg, meta, datasets, max_epochs):
    loss_fn, metric, transform, metric_name = make_task(meta)
    criterion = tabpackage.EvalCriterionConfig(
        dataset='val',
        metric=metric_name,
        metric_direction=tabpackage.MetricDirection.MINIMIZE,
    )
    model = {
        'categorical_cardinalities': meta['cat_cardinalities'],
        'd_out': 1,
        'numerical_embeddings': (
            {'type': tabpackage.nn.CosineEmbeddingsPack}
            if cfg.get('embedding_dims')
            else None
        ),
        'forward_impl': cfg.get('forward_impl', 'indexed'),
    }
    return {
        'datasets': datasets,
        'n_models': cfg['n_models'],
        'model': model,
        'prediction_transform': transform,
        'loss_fn': loss_fn,
        'optimizer': {
            'type': tabpackage.optim.MuonAdamWPack,
            'muon_skip_zero_updates': cfg.get('muon_skip_zero_updates', False),
        },
        'batch_size': cfg['batch_size'],
        'max_epochs': max_epochs,
        'early_stopping_patience': cfg['model_patience'],
        'per_model_hyperparameters': sample_hyperparameters(cfg),
        'eval_metrics': {'val': {metric_name: metric}},
        'eval_criterion': criterion,
        'eval_batch_size': cfg['inference_batch_size'],
        'ensembles': {
            'greedy': tabpackage.EnsembleConfig(
                algorithm={
                    'type': tabpackage.ensemble.greedy_ensemble,
                    'max_ensemble_size': cfg['max_ensemble_size'],
                },
                online=True,
                prediction_transform=None,
                early_stopping_patience=cfg['ensemble_patience'],
                algorithm_metric=metric,
                algorithm_metric_direction=tabpackage.MetricDirection.MINIMIZE,
                algorithm_dataset='val',
                algorithm_device=torch.device('cuda'),
            )
        },
        'ensemble_eval_metrics': {'val': {metric_name: metric}},
        'ensemble_eval_criterion': criterion,
        'stop_when_all_ensembles_are_finalized': True,
        'seed': cfg['seed'],
        'device': 'cuda',
        'aux_device': 'cpu',
        'autocast_dtype': autocast_dtype(cfg),
        # Config-level compile flags (e.g. PR #2's *_compile.json); --variant compiled
        # turns both on.
        **{k: bool(cfg.get(k, False)) for k in COMPILE_KEYS},
        'print_progress': True,
        'show_progress_bar': False,
    }


COMPILE_KEYS = (
    'compile_model',
    'compile_model_numerical_embeddings',
    'compile_model_first_block',
    'compile_optimizer_muon_newton_schulz',
)


def autocast_dtype(cfg):
    """`autocast` (common) or `autocast_dtype` (PR #1 configs): "bfloat16" or null."""
    value = cfg.get('autocast', cfg.get('autocast_dtype', 'bfloat16'))
    if value not in ('bfloat16', None):
        raise ValueError(f'autocast must be "bfloat16" or null, got {value!r}')
    return torch.bfloat16 if value else None


def setup_torch(cfg):
    """Global settings shared by all dev/pepilica scripts."""
    torch.cuda.set_device(0)
    torch.set_num_threads(cfg.get('num_threads', 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'triton': triton_version,
        'gpu': props.name,
        'compute_capability': f'{props.major}.{props.minor}',
        'gpu_uuid': str(props.uuid),
        'host': platform.node(),
    }


# >>> Inference


def run_inference(trained, arrays, cfg, inst: Instrumenter | None, split='test'):
    predictor = trained.make_predictor(
        ensemble_names=['greedy'], autocast_dtype=autocast_dtype(cfg)
    )
    predictor.cuda()
    bs = cfg['inference_batch_size']
    xn, xc = (
        torch.as_tensor(arrays[split][k], device='cuda') if k in arrays[split] else None
        for k in ('x_num', 'x_cat')
    )
    n = len(arrays[split]['y'])
    batches = [
        (None if xn is None else xn[i : i + bs], None if xc is None else xc[i : i + bs])
        for i in range(0, n, bs)
    ]
    # Exactly the schedule's steps (its wait/warmup steps are the warmup), cycling over
    # the split's batches; `--steps 0` = one full pass. Without profiling: one pass.
    if inst is not None:
        inst.schedule.start_step = 0
        inst.schedule.active = inst.schedule.active or len(batches)
    n_steps = inst.schedule.total if inst is not None else len(batches)
    tabpackage._instrument.set_instrumenter(inst)
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(n_steps):
            xn, xc = batches[i % len(batches)]
            with tabpackage._instrument.phase('infer'):
                predictor(xn, xc)
            tabpackage._instrument.step_end()
    torch.cuda.synchronize()
    tabpackage._instrument.set_instrumenter(None)
    return {
        'split': split,
        'n_rows': n,
        'n_batches': len(batches),
        'n_steps': n_steps,
        'seconds': time.perf_counter() - start,
    }


# >>> Main


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument('config', type=Path)
    p.add_argument('--mode', choices=('train', 'infer'), default='train')
    p.add_argument('--profile', choices=MODES, default='timing')
    p.add_argument(
        '--epoch',
        type=int,
        help='epoch to profile (1-based); default: config '
        'profile_epoch or 3. In infer mode: ignored',
    )
    sched = p.add_argument_group(
        'schedule (all --profile modes; same meaning as torch.profiler.schedule)',
        'Starting at --epoch: [wait, warmup, steps] repeated --repeat times. Only the '
        '--steps steps of each cycle are recorded.',
    )
    sched.add_argument(
        '--steps',
        type=int,
        default=5,
        help='recorded (active) steps per cycle; 0 = whole epoch incl. eval and epoch '
        'glue (no wait/warmup/repeat)',
    )
    sched.add_argument('--wait', type=int, default=0, help='skipped steps per cycle')
    sched.add_argument(
        '--warmup',
        type=int,
        help='run-but-not-recorded steps per cycle (default: 1, or 0 with --steps 0); '
        'the first recorded step after starting a profiler is otherwise skewed',
    )
    sched.add_argument('--repeat', type=int, default=1, help='number of cycles')
    tp = p.add_argument_group('torch.profiler options (--profile torch)')
    tp.add_argument(
        '--torch-record-shapes',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='record operator input shapes (default: on)',
    )
    tp.add_argument(
        '--torch-profile-memory',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='record tensor allocations: memory timeline in the trace (default: on)',
    )
    tp.add_argument(
        '--torch-with-stack',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='record Python stacks for every op; larger traces (default: on)',
    )
    p.add_argument(
        '--full-training',
        action='store_true',
        help="train mode: train for the config's max_epochs. By default a profiled "
        'run stops at the end of the epoch in which the schedule finishes',
    )
    p.add_argument(
        '--max-epochs',
        type=int,
        help='override config max_epochs '
        '(infer mode: epochs to train before inference; default 1)',
    )
    p.add_argument('--variant', choices=sorted(VARIANTS), default='native')
    p.add_argument(
        '--capture-phase',
        help='nvtx only: put the profiler capture range around this phase only '
        '(e.g. backward) instead of the whole window; for ncu, see README',
    )
    p.add_argument(
        '--out',
        type=Path,
        help='output dir; default .cache/runs/<name>-<mode>-<profile>-<variant>-<time>',
    )
    args = p.parse_args(argv)
    if args.warmup is None:
        args.warmup = 0 if args.steps == 0 else 1
    if args.steps == 0 and (args.wait or args.warmup or args.repeat != 1):
        p.error(
            '--steps 0 (whole epoch) cannot be combined with --wait/--warmup/--repeat'
        )
    torch_options = {
        'record_shapes': args.torch_record_shapes,
        'profile_memory': args.torch_profile_memory,
        'with_stack': args.torch_with_stack,
    }

    def make_inst(start_step):
        return Instrumenter(
            args.profile,
            out,
            Schedule(start_step, args.steps, args.wait, args.warmup, args.repeat),
            capture_phase=args.capture_phase,
            torch_options=torch_options,
        )

    if args.capture_phase and args.profile != 'nvtx':
        p.error('--capture-phase requires --profile nvtx')
    if 'NSYS_PROFILING_SESSION_ID' in os.environ and args.profile != 'nvtx':
        print(
            f'[common.run] WARNING: running under nsys with --profile {args.profile}: '
            'nsys --capture-range=cudaProfilerApi records nothing unless you use '
            '--profile nvtx',
            flush=True,
        )

    cfg = json.loads(args.config.read_text())
    name = cfg.get('name', args.config.stem)
    out = args.out or Path('.cache/runs') / (
        f'{name}-{args.mode}-{args.profile}-{args.variant}-'
        f'{datetime.datetime.now().astimezone():%Y%m%d-%H%M%S}'
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    setup_torch(cfg)

    arrays, meta = data.load(cfg)
    datasets = {
        s: {k: torch.as_tensor(v, device='cuda') for k, v in arrays[s].items()}
        for s in ('train', 'val')
    }
    steps_per_epoch = math.ceil(len(arrays['train']['y']) / cfg['batch_size'])

    if args.mode == 'train':
        epoch = args.epoch or cfg.get('profile_epoch', 3)
        train_inst = make_inst((epoch - 1) * steps_per_epoch)
        # Profiling needs training only up to the end of the epoch in which the
        # schedule finishes (the schedule may span several epochs).
        sched = train_inst.schedule
        end_step = sched.start_step + (sched.total or steps_per_epoch)
        needed_epochs = math.ceil(end_step / steps_per_epoch)
        if args.max_epochs:
            max_epochs = args.max_epochs
        elif args.profile == 'none' or args.full_training:
            max_epochs = cfg.get('max_epochs')
        else:
            max_epochs = min(needed_epochs, cfg.get('max_epochs') or needed_epochs)
        if max_epochs is not None and epoch > max_epochs:
            p.error(f'--epoch {epoch} > max_epochs {max_epochs}')
    else:
        max_epochs = args.max_epochs or 1
        train_inst = None
    infer_inst = make_inst(0) if args.mode == 'infer' else None

    kwargs = VARIANTS[args.variant](build_train_kwargs(cfg, meta, datasets, max_epochs))
    print(
        json.dumps(
            {
                'out': str(out),
                'steps_per_epoch': steps_per_epoch,
                'sizes': {s: len(a['y']) for s, a in arrays.items()},
            }
        ),
        flush=True,
    )

    tabpackage._instrument.set_instrumenter(
        train_inst if train_inst and args.profile != 'none' else None
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        trained = tabpackage.train(**kwargs)
    finally:
        tabpackage._instrument.set_instrumenter(None)
    torch.cuda.synchronize()
    report = {
        'name': name,
        'mode': args.mode,
        'variant': args.variant,
        'profile': args.profile,
        'environment': environment(),
        'steps_per_epoch': steps_per_epoch,
        'max_epochs': max_epochs,
        'train_wall_seconds': time.perf_counter() - start,
        'train_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'training_loop_seconds': trained.get_training_loop_time(),
        'training_loop_steps': trained.get_training_loop_step(),
    }
    if train_inst is not None:
        report['train_profile'] = train_inst.summary()
        if args.profile != 'none' and not train_inst.done:
            print(
                '[profiling] WARNING: training ended before the profiling window '
                'completed (lower --epoch or raise --max-epochs)',
                flush=True,
            )
    if infer_inst is not None:
        report['inference'] = run_inference(
            trained, arrays, cfg, infer_inst if args.profile != 'none' else None
        )
        report['infer_profile'] = infer_inst.summary()

    (out / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f'[common.run] wrote {out}/report.json', flush=True)
    return report


if __name__ == '__main__':
    main()
