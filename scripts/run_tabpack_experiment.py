import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Literal

from loguru import logger

import lib.env
import lib.experiment
import lib.tools.evaluate
import lib.utils
import project.tabpack

type OfflineEnsembleType = Literal['greedy']
type OnlineEnsembleType = Literal['greedy']


def _check_source_exp(source_exp: str | Path) -> Path:
    assert lib.experiment.is_experiment(source_exp)
    assert lib.experiment.is_done(source_exp)
    return Path(source_exp)


def _evaluate_ensemble(
    source_exp: str | Path,
    *,
    name: str,
    n_seeds: int,
    is_offline: bool,
    force: bool = False,
) -> None:
    """
    Retrieve ids from final ensemble (from main exp), and run method only with selected
    configs. It can be used for both online and offline ensembles.
    """

    assert n_seeds > 0

    source_exp = _check_source_exp(source_exp)
    source_config = lib.experiment.load_config(source_exp)
    source_report = lib.experiment.load_report(source_exp)

    ensemble_report = source_report['online_ensembles'][name]['report']
    ensemble_unique_ids = sorted(set(ensemble_report['ids']))

    pack_experiments = json.loads(source_exp.joinpath('experiments.json').read_text())
    # NOTE: At this moment, experiments.json and report['experiments'] are not identical
    # * Experimnt.json contains all configs (with maybe reports) in sorted way
    # * report['experiments'] contains only stopped (in order of stopping) models

    # Experiments.json guranteed to be sorted
    # Line below does not work because not all models contain report...
    # pack_experiments.sort(key=lambda x: x['report']['id'])

    base_config = deepcopy(source_config)
    if is_offline:
        assert not base_config['online_ensembles'][name][
            'include_current_ensemble_in_pool'
        ]
        assert base_config['online_ensembles'][name]['update_type'] != 'latest'
        # assert base_config['online_ensembles'][name]['patience'] > 1000, (
        # 'Ensemble patience should be infinite for offline ensemble.'
        # )

    # Adjust the config for the ensemble evaluation.
    base_config['n_models'] = len(ensemble_unique_ids)
    base_config['configs'] = [
        pack_experiments[x]['config'] for x in ensemble_unique_ids
    ]
    if base_config['optimizer']['type'] in ('AdamWPack', 'MuonAdamWPack'):
        base_config['optimizer']['shared_step'] = True

    # Remove irrelevant config fields.
    for key in list(source_config['online_ensembles'].keys()):
        if key != name:
            del base_config['online_ensembles'][key]
    for key in (
        'seed',
        'pack_size',
        'share_training_batches',
        'share_training_batch_sequence',
        'object_bagging',
        'sampler',
    ):
        base_config.pop(key, None)
        del key

    exp = (
        source_exp.parent
        / ('eval-offline-ensembles' if is_offline else 'eval-online-ensembles')
        / name
        / 'evaluation'
    )
    config = {
        'function': source_report['function'],
        'n_seeds': n_seeds,
        'base_config': base_config,
    }
    if lib.experiment.get_config_path(exp).exists():
        assert lib.experiment.load_config(exp) == config
    else:
        lib.experiment.create(exp, config=config, parents=True, force=True)
    lib.experiment.run(lib.tools.evaluate.main, None, exp, force=force, resume=True)


def main(
    exp_prefix: str | Path,
    *,
    eval_online_ensembles: list[str],
    eval_online_ensembles_n_seeds: None | int,
    eval_offline_ensembles: list[OfflineEnsembleType],
    eval_offline_ensembles_n_seeds: None | int,
    #
    clean: bool = False,
    force: bool = False,
) -> None:
    # Check the main experiment.
    exp_prefix = Path(exp_prefix)
    main_exp = exp_prefix / 'main'
    assert lib.experiment.is_experiment(main_exp)

    # If the main experiment is not done,
    # then all secondary experiments are invalid and must be removed.
    if not lib.experiment.is_done(main_exp) or force:
        for path in exp_prefix.iterdir():
            if path.is_dir() and path.name != main_exp.name:
                logger.warning(f'Removing {path}')
                shutil.rmtree(path)
            del path

    # Run the main experiment.
    lib.experiment.run(project.tabpack.main, None, main_exp, force=force)

    # Run secondary experiments.
    for online_ensemble_name in eval_online_ensembles:
        assert eval_online_ensembles_n_seeds is not None
        _evaluate_ensemble(
            main_exp,
            name=online_ensemble_name,
            n_seeds=eval_online_ensembles_n_seeds,
            is_offline=False,
            force=force,
        )

    for offline_ensemble_type in eval_offline_ensembles:
        assert eval_offline_ensembles_n_seeds is not None
        _evaluate_ensemble(
            main_exp,
            name=offline_ensemble_type,
            n_seeds=eval_offline_ensembles_n_seeds,
            is_offline=True,
            force=force,
        )
        del offline_ensemble_type

    # Finish.
    if clean:
        for directory in [
            lib.env.get_project_dir(),
            lib.env.get_snapshot_dir(),
            lib.env.get_tmp_output_dir(),
        ]:
            if directory is None:
                continue
            for path in directory.joinpath(
                exp_prefix.resolve().relative_to(lib.env.get_project_dir())
            ).rglob('*'):
                if not path.is_dir() and path.suffix in ('.pt', '.npz'):
                    path.unlink()


if __name__ == '__main__':
    lib.utils.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_prefix')
    parser.add_argument('--eval-online-ensembles', nargs='*', default=[])
    parser.add_argument('--eval-online-ensembles-n-seeds', type=int)
    parser.add_argument('--eval-offline-ensembles', nargs='*', default=[])
    parser.add_argument('--eval-offline-ensembles-n-seeds', type=int)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--force', action='store_true')

    main(**vars(parser.parse_args()))
