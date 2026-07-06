from pathlib import Path

import lib
import lib.config
import lib.datasets
import lib.env
import lib.experiment
import lib.utils


def make_config(dataset: str, *, seed: int = 0):
    return {
        'seed': seed,
        'data': lib.config.make_data_config(dataset, cache=True),
        'n_models': 64,
        'model': {
            'num_embeddings': {'type': 'CosineEmbeddingsPack'},
            'activation': 'ReLU',
            'd_block': 384,
        },
        'optimizer': {'type': 'MuonAdamWPack', 'shared_step': True},
        'batch_size': lib.config.get_batch_size(dataset),
        'n_epochs': -1,
        'patience': 16,
        'online_ensembles': {
            'greedy': {
                'type': 'greedy',
                'update_type': 'latest',
                'include_current_ensemble_in_pool': True,
                'patience': 32,
                'options': {'max_ensemble_size': 32},
            }
        },
        'sampler': {
            'type': 'RandomSampler',
            'space': {
                'model': {
                    'num_embeddings': {
                        'd_embedding': [
                            '_tune_',
                            'int',
                            8,
                            (
                                20
                                if dataset in (lib.datasets.TABRED_MAPS_ROUTING,)
                                else 32
                            ),
                            4,
                        ],
                        'init_scale': ['_tune_', 'loguniform', 0.01, 10.0],
                    },
                    'n_blocks': ['_tune_', 'int', 1, 3],
                    'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
                },
                'optimizer': {
                    'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
                    'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
                    'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
                },
            },
        },
        'amp_dtype': 'bfloat16',
        'save_all_predictions': True,
        'track_online_ensemble_history': True,
        'track_experiments': True,
    }


def main() -> None:
    commands = []
    for dataset in lib.datasets.MAIN_DATASETS:
        exp_prefix = Path(__file__).parent / dataset
        main_exp = exp_prefix / 'main'
        final_exp = exp_prefix / main_exp

        if lib.experiment.is_experiment(final_exp) and lib.experiment.is_done(
            final_exp
        ):
            # Skip fully completed experiments.
            continue

        # Do not rewrite configs of in-progress experiments.
        if not lib.experiment.is_experiment(main_exp) or lib.experiment.is_fresh(
            main_exp
        ):
            config = make_config(dataset)
            lib.experiment.create(main_exp, config=config, parents=True, force=True)

        # Save the command to run the experiment.
        command = (
            f'uv run scripts/run_tabpack_experiment.py {exp_prefix.relative_to(lib.env.get_project_dir())}'  # noqa: E501
            ' --function project.tabpack.main'
            ' --eval-online-ensembles greedy'
            ' --eval-online-ensembles-n-seeds 5'
            ' --clean'
        )
        commands.append(command)

    commands.append('')
    Path(__file__).with_name('commands.sh').write_text('\n'.join(commands))

    print('Done!')


if __name__ == '__main__':
    lib.utils.init(torch_=False)
    main()
