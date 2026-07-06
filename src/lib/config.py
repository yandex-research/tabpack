from typing import Any

import lib.datasets
import lib.env

_BATCH_SIZES = {
    lib.datasets.CHURN: 256,
    lib.datasets.CALIFORNIA: 256,
    lib.datasets.HOUSE: 256,
    lib.datasets.ADULT: 256,
    lib.datasets.DIAMOND: 512,
    lib.datasets.OTTO: 512,
    lib.datasets.HIGGS_SMALL: 512,
    lib.datasets.BLACK_FRIDAY: 512,
    lib.datasets.MICROSOFT: 1024,
    #
    lib.datasets.TABRED_SBERBANK_HOUSING: 256,
    lib.datasets.TABRED_ECOM_OFFERS: 1024,
    lib.datasets.TABRED_MAPS_ROUTING: 1024,
    lib.datasets.TABRED_HOMESITE_INSURANCE: 1024,
    lib.datasets.TABRED_COOKING_TIME: 1024,
    lib.datasets.TABRED_HOMECREDIT_DEFAULT: 1024,
    lib.datasets.TABRED_DELIVERY_ETA: 1024,
    lib.datasets.TABRED_WEATHER: 1024,
}


def get_batch_size(dataset: str) -> int:
    return _BATCH_SIZES[dataset]


_NUM_TRANSFORMS = {
    x: 'noisy-quantile'
    for x in _BATCH_SIZES.keys()
    if x
    not in {
        # The "noisy-quantile" normalization works poorly for the OTTO dataset.
        lib.datasets.OTTO,
        # The following TabReD datasets are already normalized.
        lib.datasets.TABRED_MAPS_ROUTING,
        lib.datasets.TABRED_COOKING_TIME,
        lib.datasets.TABRED_DELIVERY_ETA,
    }
}


def get_num_transform(dataset: str) -> str | None:
    return _NUM_TRANSFORMS.get(dataset)


def make_data_config(dataset: str, *, cache: bool = False) -> dict[str, Any]:
    data_config = {}

    dataset_path = lib.env.get_data_dir() / dataset
    if not dataset_path.exists():
        raise RuntimeError(f'The dataset does not exist: {dataset_path}')
    data_config['path'] = str(dataset_path.relative_to(lib.env.get_project_dir()))

    data_config['extract_bin_from_num'] = True

    num_transform = _NUM_TRANSFORMS.get(dataset)
    if num_transform is not None:
        data_config['num_policy'] = num_transform
    data_config['bin_policy'] = 'convert-to-cat'
    if dataset_path.joinpath('x_cat.npy').exists():
        data_config['cat_policy'] = 'ordinal'

    data_config['cache'] = cache

    return data_config
