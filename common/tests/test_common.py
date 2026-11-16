"""Tests for common/ (needs a CUDA GPU; run inside the docker image):

docker/run.sh pytest common/tests -q
"""

import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tabpackage
import tabpackage._instrument
from common import data, run
from common.profiling import Schedule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

CFG = json.loads(Path('common/configs/synthetic_small.json').read_text())
CFG.update(
    max_epochs=2,
    profile_epoch=2,
    n_models=4,
    synthetic={
        'n_train': 4096,
        'n_val': 1024,
        'n_test': 1024,
        'n_num': 8,
        'n_cat': 2,
        'cardinality': 4,
    },
)
STEP_PHASES = ['gather', 'forward', 'loss', 'backward', 'optimizer', 'zero_grad']


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(CFG))
    return p


def _train(cfg, instrumenter=None):
    run.setup_torch(cfg)
    arrays, meta = data.load(cfg)
    ds = {
        s: {k: torch.as_tensor(v, device='cuda') for k, v in arrays[s].items()}
        for s in ('train', 'val')
    }
    kwargs = run.build_train_kwargs(cfg, meta, ds, cfg['max_epochs'])
    kwargs['print_progress'] = False
    tabpackage._instrument.set_instrumenter(instrumenter)
    try:
        return tabpackage.train(**kwargs)
    finally:
        tabpackage._instrument.set_instrumenter(None)


class Recorder:
    def __init__(self):
        self.calls = []

    def phase(self, name):
        self.calls.append(name)
        return contextlib.nullcontext()

    def step_end(self):
        self.calls.append('step_end')

    def epoch_end(self):
        self.calls.append('epoch_end')


def test_phase_order_and_no_behaviour_change():
    rec = Recorder()
    with_hooks = _train(copy.deepcopy(CFG), rec)
    without = _train(copy.deepcopy(CFG), None)

    steps_per_epoch = 4096 // CFG['batch_size']
    one_epoch = [*STEP_PHASES, 'step_end'] * steps_per_epoch + ['eval', 'epoch_end']
    assert rec.calls == one_epoch * CFG['max_epochs']

    # Same seed -> bit-identical metrics with and without hooks (wall time excluded).
    h1, h2 = with_hooks.load_model_history(), without.load_model_history()
    assert len(h1) == len(h2) == CFG['max_epochs']
    for a, b in zip(h1, h2):
        assert torch.equal(a['metrics']['val']['rmse'], b['metrics']['val']['rmse'])
        assert torch.equal(a['steps'], b['steps'])


@pytest.mark.parametrize('mode', ['timing', 'torch', 'nvtx', 'memory'])
def test_train_modes(cfg_path, tmp_path, mode):
    out = tmp_path / mode
    rep = run.main(
        [str(cfg_path), '--profile', mode, '--steps', '3', '--out', str(out)]
    )
    assert (out / 'report.json').is_file()
    prof = rep['train_profile']
    if mode == 'timing':
        assert set(STEP_PHASES) <= prof['phase_ms'].keys()
        assert all(v['count'] == 3 for v in prof['phase_ms'].values())
        assert 'eval' not in prof['phase_ms']  # the window must not open at the eval
    if mode == 'memory':
        assert set(STEP_PHASES) <= prof['phase_memory_mib'].keys()
    if mode == 'torch':
        trace = json.loads((out / 'trace.json').read_text())
        names = {e.get('name') for e in trace['traceEvents']}
        assert {f'tabpack/{p}' for p in STEP_PHASES} <= names


def test_whole_epoch_window_includes_eval(cfg_path, tmp_path):
    rep = run.main(
        [
            str(cfg_path),
            '--profile',
            'timing',
            '--steps',
            '0',
            '--out',
            str(tmp_path / 'o'),
        ]
    )
    assert 'eval' in rep['train_profile']['phase_ms']


def test_infer_mode(cfg_path, tmp_path):
    rep = run.main(
        [
            str(cfg_path),
            '--mode',
            'infer',
            '--profile',
            'timing',
            '--steps',
            '2',
            '--out',
            str(tmp_path / 'o'),
        ]
    )
    assert rep['infer_profile']['phase_ms']['infer']['count'] == 2
    assert rep['inference']['n_rows'] == 1024


@pytest.mark.parametrize('mode', ['torch', 'nvtx', 'memory'])
def test_infer_modes(cfg_path, tmp_path, mode):
    out = tmp_path / mode
    rep = run.main(
        [
            str(cfg_path),
            '--mode',
            'infer',
            '--profile',
            mode,
            '--steps',
            '2',
            '--out',
            str(out),
        ]
    )
    assert rep['inference']['n_batches'] == 1
    if mode == 'torch':
        names = {
            e.get('name')
            for e in json.loads((out / 'trace.json').read_text())['traceEvents']
        }
        assert 'tabpack/infer' in names
    if mode == 'memory':
        assert len(rep['infer_profile']['phase_memory_mib']['infer']) == 3


def test_binclass(tmp_path):
    cfg = {**CFG, 'synthetic': {**CFG['synthetic'], 'task': 'binclass'}}
    arrays, meta = data.load(cfg)
    assert meta['task'] == 'binclass'
    assert set(arrays['train']['y'].tolist()) == {0.0, 1.0}
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(cfg))
    rep = run.main(
        [str(p), '--mode', 'infer', '--profile', 'timing', '--out', str(tmp_path / 'o')]
    )
    assert rep['training_loop_steps'] > 0


def test_compiled_variant(cfg_path, tmp_path):
    rep = run.main(
        [
            str(cfg_path),
            '--variant',
            'compiled',
            '--steps',
            '2',
            '--out',
            str(tmp_path / 'o'),
        ]
    )
    assert rep['train_profile']['phase_ms']['forward']['count'] == 2


def _write_raw_dataset(d, *, n=3000, n_num=8, task='regression', seed=0):
    """A directory in the TabReD / tabpack-data layout (fixture: synthetic values)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    (d / 'splits' / 'default').mkdir(parents=True)
    x_num = rng.normal(size=(n, n_num)).astype(np.float32)
    x_num[::50, 0] = np.nan  # real TabReD data has missing values
    y = x_num[:, 1] + rng.normal(size=n)
    np.save(d / 'x_num.npy', x_num)
    np.save(d / 'y.npy', (y > 0).astype(np.int64) if task == 'binclass' else y)
    if task == 'binclass':
        # As in the real TabReD files: binary features are float 0/1 with missing
        # values (NaN), categorical features are int64 codes.
        x_bin = rng.integers(0, 2, (n, 2)).astype(np.float32)
        x_bin[::40, 1] = np.nan
        np.save(d / 'x_bin.npy', x_bin)
        np.save(d / 'x_cat.npy', rng.integers(0, 3, (n, 3)))
    # The real TabReD info.json format.
    (d / 'info.json').write_text(json.dumps({'task': {'type': task, 'score': 'rmse'}}))
    perm = rng.permutation(n)
    for name, idx in zip(
        ('train', 'val', 'test'), np.split(perm, [n * 6 // 10, n * 8 // 10])
    ):
        np.save(d / 'splits' / 'default' / f'{name}.npy', idx)


def test_california_config_with_embeddings(tmp_path):
    """California config (cosine numerical embeddings) on a raw-directory fixture
    with California's shape (8 numeric features, regression)."""
    _write_raw_dataset(tmp_path / 'data' / 'raw-fixture')
    cfg = json.loads(Path('common/configs/california.json').read_text())
    cfg.update(
        dataset='raw-fixture',
        data_root=str(tmp_path / 'data'),
        n_models=4,
        max_epochs=2,
        profile_epoch=2,
        batch_size=256,
    )
    arrays, meta = data.load(cfg)
    assert meta['n_num'] == 8 and meta['cat_cardinalities'] == []
    assert all(np.isfinite(arrays[s]['x_num']).all() for s in data.SPLITS)
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(cfg))
    rep = run.main(
        [str(p), '--profile', 'timing', '--steps', '2', '--out', str(tmp_path / 'o')]
    )
    assert rep['train_profile']['phase_ms']['forward']['count'] == 2


def test_tabred_raw_dir_binclass(tmp_path):
    """TabReD directory in the real format: info.json task, NaN binary, int codes."""
    _write_raw_dataset(
        tmp_path / 'data' / 'tabred' / 'homesite-insurance', task='binclass'
    )
    cfg = {**CFG, 'dataset': 'homesite-insurance', 'data_root': str(tmp_path / 'data')}
    arrays, meta = data.load(cfg)
    # The task comes from info.json's {"task": {"type": ...}} (it used to be missed,
    # silently training binclass datasets as regression).
    assert meta['task'] == 'binclass' and meta['n_num'] == 8
    cards = np.array(meta['cat_cardinalities'])
    assert len(cards) == 5  # 2 binary + 3 categorical
    assert cards[0] == 3 and cards[2:].tolist() == [4, 4, 4]  # values + unknown
    assert cards[1] == 4  # 0, 1, NaN (as a category) + unknown
    for s in data.SPLITS:
        x_cat = arrays[s]['x_cat']
        assert (x_cat >= 0).all() and (x_cat < cards).all()
        assert set(np.unique(arrays[s]['y'])) <= {0.0, 1.0}
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(cfg))
    run.main([str(p), '--profile', 'none', '--out', str(tmp_path / 'o')])


def test_paper_cache_layout(tmp_path, monkeypatch):
    """`.cache/<name>-paper`, as written by preprocess_maps_paper.py."""
    import numpy as np

    cfg_text = json.dumps({**CFG, 'dataset': 'maps-routing'})
    monkeypatch.chdir(tmp_path)
    d = tmp_path / '.cache' / 'maps-routing-paper'
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, n in (('train', 2048), ('val', 512), ('test', 512)):
        np.save(d / f'{split}_x_num.npy', rng.normal(size=(n, 6)).astype(np.float32))
        np.save(d / f'{split}_x_cat.npy', rng.integers(0, 5, (n, 2)))
        np.save(d / f'{split}_y.npy', rng.normal(size=n).astype(np.float32))
    (d / 'metadata.json').write_text(
        json.dumps(
            {
                'n_num': 6,
                'cat_cardinalities': [5, 5],
                'target_mean': 0.0,
                'target_std': 2.5,
            }
        )
    )
    _, meta = data.load(json.loads(cfg_text))
    assert meta == {
        'n_num': 6,
        'cat_cardinalities': [5, 5],
        'task': 'regression',
        'target_std': 2.5,
    }
    (tmp_path / 'cfg.json').write_text(cfg_text)
    run.main(['cfg.json', '--profile', 'timing', '--steps', '2', '--out', 'o'])


def test_capture_phase(cfg_path, tmp_path):
    """--capture-phase runs and is validated (ncu itself: checked by hand)."""
    run.main(
        [
            str(cfg_path),
            '--profile',
            'nvtx',
            '--steps',
            '2',
            '--capture-phase',
            'backward',
            '--out',
            str(tmp_path / 'o'),
        ]
    )
    with pytest.raises(SystemExit):
        run.main(
            [
                str(cfg_path),
                '--profile',
                'timing',
                '--capture-phase',
                'backward',
                '--out',
                str(tmp_path / 'o2'),
            ]
        )


@pytest.mark.parametrize(
    'path', sorted(Path('common/configs').glob('*.json')), ids=lambda p: p.stem
)
def test_every_config_builds(path):
    """Every config in common/configs gives valid train kwargs."""
    cfg = json.loads(path.read_text())
    meta = {'n_num': 4, 'cat_cardinalities': [3], 'task': 'regression', 'target_std': 1}
    kwargs = run.build_train_kwargs(cfg, meta, {}, cfg.get('max_epochs'))
    hparams = kwargs['per_model_hyperparameters']
    assert len(hparams) == cfg['n_models']
    assert {h['model']['d_block'] for h in hparams} <= set(cfg['widths'])
    assert kwargs['compile_model'] == cfg.get('compile_model', False)
    assert kwargs['autocast_dtype'] == torch.bfloat16
    if cfg.get('embedding_dims'):
        assert all('numerical_embeddings' in h['model'] for h in hparams)


def test_npz_dataset(tmp_path):
    """A single .npz with `<split>.<key>` arrays (`data` key), plus the `task_type`
    and `autocast_dtype` config keys."""
    rng = np.random.default_rng(0)
    arrays = {}
    for split, n in (('train', 2048), ('val', 512), ('test', 512)):
        arrays[f'{split}.x_num'] = rng.normal(size=(n, 5)).astype(np.float32)
        arrays[f'{split}.x_cat'] = rng.integers(0, 7, (n, 3))
        arrays[f'{split}.y'] = rng.integers(0, 2, n)
    np.savez(tmp_path / 'dataset.npz', **arrays)
    cfg = {
        **CFG,
        'dataset': 'not-used-when-data-exists',
        'data': str(tmp_path / 'dataset.npz'),
        'data_root': str(tmp_path),  # keep the preprocessed cache out of the repo
        'task_type': 'binclass',
        'autocast_dtype': 'bfloat16',
    }
    _, meta = data.load(cfg)
    assert meta['task'] == 'binclass' and meta['n_num'] == 5
    assert len(meta['cat_cardinalities']) == 3
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(cfg))
    rep = run.main(
        [str(p), '--profile', 'timing', '--steps', '2', '--out', str(tmp_path / 'o')]
    )
    assert rep['train_profile']['phase_ms']['forward']['count'] == 2


def test_schedule_cycles():
    s = Schedule(10, active=3, wait=1, warmup=2, repeat=2)
    where = [s.where(10 + i) for i in range(s.total)]
    assert [kind for _, kind in where] == ['wait', 'warmup', 'warmup'] + [
        'active'
    ] * 3 + ['wait', 'warmup', 'warmup'] + ['active'] * 3
    assert [k for k, _ in where] == [0] * 6 + [1] * 6
    with pytest.raises(ValueError):
        Schedule(0, active=0, wait=1)  # whole epoch can't have wait/warmup/repeat


# 4096 rows / batch 512 = 8 steps per epoch; epoch 2 = steps 8..15, and the schedule
# below covers exactly those 8 steps: 2 x (1 wait + 1 warmup + 2 active).
SCHEDULE = ['--wait', '1', '--warmup', '1', '--steps', '2', '--repeat', '2']


def test_timing_schedule(cfg_path, tmp_path):
    rep = run.main(
        [str(cfg_path), '--profile', 'timing', *SCHEDULE, '--out', str(tmp_path / 'o')]
    )
    prof = rep['train_profile']
    assert all(v['count'] == 4 for v in prof['phase_ms'].values())  # 2 cycles x 2
    assert [c['forward']['count'] for c in prof['phase_ms_per_cycle']] == [2, 2]


def test_torch_schedule_one_trace_per_cycle(cfg_path, tmp_path):
    out = tmp_path / 'o'
    rep = run.main(
        [
            str(cfg_path),
            '--profile',
            'torch',
            *SCHEDULE,
            '--no-torch-record-shapes',
            '--out',
            str(out),
        ]
    )
    assert rep['train_profile']['traces'] == 2
    for k in (0, 1):
        events = json.loads((out / f'trace-{k}.json').read_text())['traceEvents']
        forwards = [
            e
            for e in events
            if e.get('name') == 'tabpack/forward' and e.get('cat') == 'user_annotation'
        ]
        assert len(forwards) == 2  # only the active steps, not wait/warmup
        assert (out / f'operators-{k}.txt').is_file()


def test_nvtx_schedule_runs(cfg_path, tmp_path):
    """Capture ranges per active block (checked under nsys by hand, see README)."""
    rep = run.main(
        [str(cfg_path), '--profile', 'nvtx', *SCHEDULE, '--out', str(tmp_path / 'o')]
    )
    assert rep['train_profile']['schedule']['repeat'] == 2


CALIFORNIA_FILE = Path('data/california-sklearn/cal_housing_py3.pkz')


@pytest.mark.skipif(
    not CALIFORNIA_FILE.is_file(),
    reason='California Housing not downloaded yet (first `california-sklearn` run '
    'downloads it)',
)
def test_california_sklearn_split():
    """All California configs: scikit-learn data with PR #2's seeded 64/16/20 split."""
    for f in sorted(Path('common/configs').glob('california*.json')):
        assert json.loads(f.read_text())['dataset'] == 'california-sklearn', f.name
    arrays, meta = data.load({'dataset': 'california-sklearn', 'seed': 42})
    assert {s: len(arrays[s]['y']) for s in data.SPLITS} == {
        'train': 13209,
        'val': 3303,
        'test': 4128,
    }
    assert meta['n_num'] == 8 and meta['task'] == 'regression'


def test_profiled_run_stops_after_schedule(tmp_path):
    """No max_epochs in the config: profiling epoch 2 trains exactly 2 epochs."""
    cfg = {**CFG, 'max_epochs': None}
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(cfg))
    rep = run.main(
        [
            str(p),
            '--profile',
            'timing',
            '--epoch',
            '2',
            '--steps',
            '2',
            '--out',
            str(tmp_path / 'o'),
        ]
    )
    assert rep['max_epochs'] == 2
    assert rep['training_loop_steps'] == 2 * 8  # 4096 rows / batch 512
    assert rep['train_profile']['phase_ms']['forward']['count'] == 2


def test_preprocessed_cache(tmp_path, capsys):
    """Second load comes from <data_root>/preprocessed; source changes rebuild it."""
    d = tmp_path / 'data' / 'tabred' / 'homesite-insurance'
    _write_raw_dataset(d, task='binclass')
    cfg = {
        'dataset': 'homesite-insurance',
        'data_root': str(tmp_path / 'data'),
        'seed': 0,
    }

    a1, m1 = data.load(cfg)
    assert 'cached preprocessed' in capsys.readouterr().out
    cache = tmp_path / 'data' / 'preprocessed'
    assert (cache / 'homesite-insurance-seed0.npz').is_file()
    assert (cache / 'homesite-insurance-seed0.json').is_file()

    a2, m2 = data.load(cfg)
    assert 'loaded preprocessed' in capsys.readouterr().out
    assert m1 == m2
    for s in data.SPLITS:
        for k in a1[s]:
            np.testing.assert_array_equal(a1[s][k], a2[s][k])

    # A changed source file (e.g. a re-download) invalidates the cache.
    y = np.load(d / 'y.npy')
    np.save(d / 'y.npy', 1 - y)
    a3, _ = data.load(cfg)
    assert 'cached preprocessed' in capsys.readouterr().out
    np.testing.assert_array_equal(a3['train']['y'], 1 - a1['train']['y'])

    # Opt-out.
    data.load({**cfg, 'cache_preprocessed': False})
    assert 'preprocessed' not in capsys.readouterr().out
