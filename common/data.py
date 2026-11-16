"""Dataset loading for common.run.

`load(cfg)` returns ``(arrays, meta)`` where ``arrays[split] = {'x_num', 'x_cat', 'y'}``
(numpy, any key may be missing) for splits ``train``/``val``/``test``, and ``meta`` has
``n_num``, ``cat_cardinalities``, ``task`` (``regression`` | ``binclass``) and
``target_std``.

If ``cfg['data']`` points to an existing ``.npz`` (``<split>.<key>`` arrays, the
format of PR #1's dev/nosorozhek/scripts/prepare_tabred.py), it is used and
``cfg['task_type']`` gives the task. Otherwise ``cfg['dataset']`` selects (see
common/README.md, "Data"):

* ``synthetic``   generated on the fly, no files needed (``synthetic.task``:
  ``regression`` or ``binclass``);
* ``california-sklearn``  scikit-learn's California Housing, seeded random 64/16/20
  split (downloaded automatically into ``data/california-sklearn/``; used by all
  California configs);
* any TabReD name, e.g. ``maps-routing``, ``homecredit-default``, ``ecom-offers``,
  ``cooking-time``, ``delivery-eta``, ``homesite-insurance``, ``sberbank-housing``,
  ``weather``. Looked up, in order, as
  1. a ready cache ``.cache/<name>-paper/{metadata.json,<split>_{x_num,x_cat,y}.npy}``
     (the format produced by dev/pepilica/scripts/preprocess_maps_paper.py), then
  2. a raw directory ``<data_root>/tabred/<name>`` (where ``tabred download`` puts it)
     or ``<data_root>/<name>`` with ``{x_num,x_bin,x_cat,y}.npy``, ``info.json`` and
     ``splits/default/*.npy``.

``<data_root>`` is ``cfg['data_root']`` if set, else ``data/`` (repo root, gitignored),
then ``.cache/data/``.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test')
DEFAULT_DATA_ROOTS = [Path('data'), Path('.cache/data')]
# Bump when `_preprocess` changes: invalidates every preprocessed cache.
PREPROCESS_VERSION = 1
# Datasets downloadable with Yandex's `tabred` tool (see common/README.md, "Data").
TABRED = (
    'cooking-time',
    'delivery-eta',
    'ecom-offers',
    'homecredit-default',
    'homesite-insurance',
    'maps-routing',
    'sberbank-housing',
    'weather',
)
PAPER_CACHE_ALIASES = {
    'homecredit-default': 'homecredit',
    'maps-routing': 'maps-routing',
}


def load(cfg: dict) -> tuple[dict, dict]:
    name = cfg['dataset']
    seed = cfg.get('seed', 0)
    # Preprocessed TabReD / .npz data is cached in <data_root>/preprocessed/.
    cache_dir = (
        (Path(cfg['data_root']) if cfg.get('data_root') else DEFAULT_DATA_ROOTS[0])
        / 'preprocessed'
        if cfg.get('cache_preprocessed', True)
        else None
    )
    if cfg.get('data') and Path(cfg['data']).is_file():
        path = Path(cfg['data'])
        key = (
            f'{path.stem}-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]}'
        )
        return _cached(
            cache_dir,
            f'{key}-seed{seed}',
            [path],
            lambda: _npz(path, cfg.get('task_type'), seed),
        )
    if name == 'synthetic':
        return _synthetic(cfg.get('synthetic', {}), cfg.get('seed', 0))
    if name == 'california-sklearn':
        root = Path(cfg['data_root']) if cfg.get('data_root') else DEFAULT_DATA_ROOTS[0]
        return _california_sklearn(root / 'california-sklearn', cfg.get('seed', 0))

    # `data_root` from the config, else `data` (repo root), then `.cache/data`.
    roots = [Path(cfg['data_root'])] if cfg.get('data_root') else DEFAULT_DATA_ROOTS
    data_root = roots[0]
    paper = Path('.cache') / f'{PAPER_CACHE_ALIASES.get(name, name)}-paper'
    if (paper / 'metadata.json').is_file():
        return _paper_cache(paper)

    dirs = [d for r in roots for d in (r / 'tabred' / name, r / name)]
    for d in dirs:
        if (d / 'y.npy').is_file():
            sources = sorted(
                f
                for f in [
                    *d.glob('*.npy'),
                    d / 'info.json',
                    *d.glob('splits/default/*'),
                ]
                if f.is_file()
            )
            return _cached(
                cache_dir, f'{name}-seed{seed}', sources, lambda d=d: _raw_dir(d, seed)
            )

    checked = [
        *([cfg['data']] if cfg.get('data') else []),
        f'{paper}/metadata.json',
        *(f'{d}/y.npy' for d in dirs),
    ]
    hint = (
        f'\nDownload it once (needs a Kaggle API token in ~/.kaggle/kaggle.json), from '
        f'the repo root on the host:\n  uvx --with six tabred download {name} '
        f'--output-path {data_root / "tabred"}'
        if name in TABRED
        else ''
    )
    raise FileNotFoundError(
        f'Dataset {name!r} is not on this machine. Looked for:\n  '
        + '\n  '.join(checked)
        + hint
        + '\nSee common/README.md, section "Data" (no-download alternatives: '
        'dataset "synthetic" or "california-sklearn").'
    )


def _cached(cache_dir: Path | None, key: str, sources: list[Path], build):
    """Return `build()` (preprocessed arrays + meta), cached in `cache_dir/key.npz`.

    The cache is reused only if its signature matches: PREPROCESS_VERSION and the
    path, size and mtime of every source file (re-downloading the data or changing the
    preprocessing rebuilds it). Arrays are written to a temp file and renamed, and the
    signature file is written last, so a half-written cache is never read."""
    if cache_dir is None:
        return build()
    signature = {
        'version': PREPROCESS_VERSION,
        'sources': {
            str(f): [f.stat().st_size, f.stat().st_mtime_ns] for f in sorted(sources)
        },
    }
    arrays_file, meta_file = cache_dir / f'{key}.npz', cache_dir / f'{key}.json'
    if arrays_file.is_file() and meta_file.is_file():
        stored = json.loads(meta_file.read_text())
        if stored.get('signature') == signature:
            with np.load(arrays_file) as z:
                arrays = {
                    s: {
                        k.split('.', 1)[1]: z[k]
                        for k in z.files
                        if k.startswith(f'{s}.')
                    }
                    for s in SPLITS
                }
            print(f'[data] loaded preprocessed {key} from {arrays_file}', flush=True)
            return arrays, stored['meta']

    arrays, meta = build()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f'.{key}.{os.getpid()}.tmp.npz'
    np.savez(
        tmp, **{f'{s}.{k}': v for s, part in arrays.items() for k, v in part.items()}
    )
    tmp.replace(arrays_file)
    tmp_meta = meta_file.with_name(f'.{key}.{os.getpid()}.tmp.json')
    tmp_meta.write_text(json.dumps({'signature': signature, 'meta': meta}, indent=1))
    tmp_meta.replace(meta_file)
    print(f'[data] cached preprocessed {key} in {arrays_file}', flush=True)
    return arrays, meta


def _california_sklearn(data_home: Path, seed: int):
    """California Housing from scikit-learn (downloaded once into
    ``data/california-sklearn/``) with
    the seeded random 64/16/20 split of PR #2's run_profiling_experiment.py.
    Not the tabpack-data archive's split used by dev/pepilica: same 20640 rows and 8
    features, different train/val/test rows."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    raw = fetch_california_housing(data_home=data_home)
    rows = np.arange(len(raw.target))
    trainval, test = train_test_split(rows, test_size=0.2, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.2, random_state=seed)
    x, y = raw.data.astype(np.float32), raw.target.astype(np.float32)
    parts = {
        s: {'x_num': x[i], 'y': y[i]}
        for s, i in (('train', train), ('val', val), ('test', test))
    }
    return _preprocess(parts, 'regression', seed)


def _synthetic(s: dict, seed: int):
    rng = np.random.default_rng(seed)
    n = {
        'train': s.get('n_train', 65536),
        'val': s.get('n_val', 8192),
        'test': s.get('n_test', 8192),
    }
    n_num, n_cat, card = s.get('n_num', 32), s.get('n_cat', 4), s.get('cardinality', 8)
    w = rng.normal(size=n_num).astype(np.float32)
    arrays = {}
    for split, size in n.items():
        x_num = rng.normal(size=(size, n_num)).astype(np.float32)
        x_cat = rng.integers(0, card, size=(size, n_cat), dtype=np.int64)
        y = np.tanh(x_num @ w / np.sqrt(n_num)) + 0.1 * (x_cat.sum(1) % 3)
        arrays[split] = {'x_num': x_num, 'y': y.astype(np.float32)}
        if n_cat:
            arrays[split]['x_cat'] = x_cat
    task = s.get('task', 'regression')
    if task == 'binclass':  # threshold at the train median -> balanced 0/1 labels
        threshold = float(np.median(arrays['train']['y']))
        for a in arrays.values():
            a['y'] = (a['y'] > threshold).astype(np.float32)
        std = 1.0
    else:
        y_tr = arrays['train']['y']
        mean, std = float(y_tr.mean()), float(y_tr.std())
        for a in arrays.values():
            a['y'] = (a['y'] - mean) / std
    meta = {
        'n_num': n_num,
        'cat_cardinalities': [card] * n_cat,
        'task': task,
        'target_std': std,
    }
    return arrays, meta


def _paper_cache(d: Path):
    meta = json.loads((d / 'metadata.json').read_text())
    arrays = {
        split: {
            k: np.load(d / f'{split}_{k}.npy', allow_pickle=False)
            for k in ('x_num', 'x_cat', 'y')
            if (d / f'{split}_{k}.npy').is_file()
        }
        for split in SPLITS
    }
    y = arrays['train']['y']
    task = 'binclass' if np.isin(np.unique(y), [0, 1]).all() else 'regression'
    return arrays, {
        'n_num': meta['n_num'],
        'cat_cardinalities': meta['cat_cardinalities'],
        'task': task,
        'target_std': float(meta.get('target_std') or 1.0),
    }


def _raw_dir(d: Path, seed: int):
    """Generic TabReD-style directory (see module docstring)."""
    info = (
        json.loads((d / 'info.json').read_text()) if (d / 'info.json').is_file() else {}
    )
    idx = {s: np.load(d / 'splits' / 'default' / f'{s}.npy') for s in SPLITS}
    full = {
        k: np.load(d / f'{k}.npy', allow_pickle=True)
        for k in ('x_num', 'x_bin', 'x_cat', 'y')
        if (d / f'{k}.npy').is_file()
    }
    parts = {s: {k: v[idx[s]] for k, v in full.items()} for s in SPLITS}
    # TabReD: {"task": {"type": "binclass", ...}}; older layouts: {"task_type": ...}.
    task = (info.get('task') or {}).get('type') or info.get('task_type', 'regression')
    return _preprocess(parts, task, seed)


def _npz(path: Path, task: str | None, seed: int):
    """One .npz with ``<split>.<key>`` arrays, as written by PR #1's
    dev/nosorozhek/scripts/prepare_tabred.py (``data/profiling/<name>.npz``)."""
    with np.load(path, allow_pickle=True) as z:
        parts = {
            s: {
                k: z[f'{s}.{k}']
                for k in ('x_num', 'x_bin', 'x_cat', 'y')
                if f'{s}.{k}' in z.files
            }
            for s in SPLITS
        }
    if task is None:
        y = parts['train']['y']
        task = 'binclass' if np.isin(np.unique(y), [0, 1]).all() else 'regression'
    return _preprocess(parts, task, seed)


def _preprocess(parts: dict, task: str, seed: int):
    """Fitted on train only: quantile-normal numerics (NaN -> 0), binary and
    categorical features -> ordinal codes (+1 'unknown' category), standardized
    regression target."""
    from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

    if task not in ('regression', 'binclass'):
        raise NotImplementedError(f'task_type={task!r} is not supported yet')
    out = {s: {} for s in SPLITS}
    meta = {'n_num': 0, 'cat_cardinalities': [], 'task': task, 'target_std': 1.0}

    if 'x_num' in parts['train']:
        x = {s: np.nan_to_num(parts[s]['x_num'].astype(np.float32)) for s in SPLITS}
        tr = x['train']
        qt = QuantileTransformer(
            n_quantiles=max(min(len(tr) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
            random_state=seed,
        ).fit(tr + np.random.default_rng(seed).normal(0, 1e-5, tr.shape))
        for s in SPLITS:
            out[s]['x_num'] = qt.transform(x[s]).astype(np.float32)
        meta['n_num'] = tr.shape[1]

    cat_keys = [k for k in ('x_bin', 'x_cat') if k in parts['train']]
    if cat_keys:
        # Numeric codes (TabReD) are encoded as numbers (NaN = missing); anything
        # else (e.g. strings) as strings.
        numeric = all(
            np.issubdtype(parts['train'][k].dtype, np.number) for k in cat_keys
        )
        dtype = np.float64 if numeric else str
        x = {
            s: np.concatenate([parts[s][k].astype(dtype) for k in cat_keys], 1)
            for s in SPLITS
        }
        enc = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int64,
        ).fit(x['train'])
        cards = [len(c) + 1 for c in enc.categories_]  # +1: unknown category
        for s in SPLITS:
            v = enc.transform(x[s])
            out[s]['x_cat'] = np.where(v < 0, np.array(cards) - 1, v)
        meta['cat_cardinalities'] = cards

    y = {s: parts[s]['y'].astype(np.float32).reshape(-1) for s in SPLITS}
    if task == 'regression':
        mean, std = float(y['train'].mean()), float(y['train'].std())
        y = {s: (v - mean) / std for s, v in y.items()}
        meta['target_std'] = std
    for s in SPLITS:
        out[s]['y'] = y[s]
    return out, meta
