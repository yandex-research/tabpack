"""TabPack training example.

TabPack represents a set of MLPs with different hyperparameters trained efficiently in
parallel with on-the-fly ensemble construction, for supervised learning on tabular data.
In one TabPack training run, many MLPs and ensembles can be trained and built.

The central concept behind TabPack is _model pack_. A model pack is a set of models of
the same family (MLPs in the case of TabPack) with different hyperparameters, stacked
together in one module. They are trained with _optimizer packs_: optimizers with
different hyperparameters implemented as one large optimizer. This approach enables
efficient training of many (tens, hundreds, thousands, ...) different models in
parallel. To train more models than fits into memory, one can limit the number of models
trained in parallel.

While the models are training, their intermediate and/or final checkpoints are used to
build ensembles. Since ensemble construction is cheap compared to model training, one
can build many ensembles in one run and then choose the one that works best.
"""

import datetime
import itertools
import math
import os
import random
from functools import partial
from pathlib import Path

# Make the code runnable on Apple Silicon for PyTorch <2.12.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn  # noqa: PLR0402
from torch import Tensor

import tabpackage  # TabPack training
import tabpackage.ensemble  # Ensemble algorithms
import tabpackage.metrics  # Metric computation for model packs
import tabpackage.nn  # Module packs
import tabpackage.optim  # Optimizer packs


def make_dataset() -> tuple[dict[str, np.ndarray], list[int]]:
    """Make an example of a tabular dataset."""

    # Generate numerical features and labels.
    X_num, Y = sklearn.datasets.make_classification(
        n_samples=20000,
        n_features=8,
        # n_classes=2,  # Binary classification
        n_classes=3,  # Multiclass classification
        n_informative=3,
        n_redundant=2,
    )

    # Generate random categorical features for demonstration purposes.
    categorical_cardinalities = [
        4,  # Allowed values: [0, 1, 2, 3].
        7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
    ]
    X_cat = np.column_stack(
        # For `tabpackage.nn.TabPack`, for a categorical feature with the number of
        # unique values `c`, its in-vocabulary values are expected to be from
        # `range(c)`, and its out-of-vocabulary ("unknown") values are expected to equal
        # exactly `c`. Thus, the following code samples values from `range(c + 1)` to
        # emulate both known and unknown values.
        [np.random.randint(0, c + 1, (len(Y),)) for c in categorical_cardinalities]
    )

    dataset = {
        # The `tabpackage.train` function requires specific set of keys and data types.
        'x_num': X_num.astype(np.float32),
        'x_cat': X_cat.astype(np.int64),
        'y': Y.astype(np.int64),
    }

    return dataset, categorical_cardinalities


def split_dataset(dataset: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Split a dataset into the training, validation and test datasets."""
    ratio = 0.8

    all_idx = np.arange(len(dataset['y']))
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=ratio
    )

    # Using additional dataset splits, most notably a validation split for early
    # stopping to avoid overfitting, is a common pattern in machine learning. TabPack,
    # however, needs an additional split not only for stopping, but also for building
    # ensembles that are usually used to make predictions after the training. This is
    # why:
    #
    # - For the _first_ training run of TabPack with ensembles, an additional dataset
    #   split is _required_ and is not optional. The typical baseline is to use the same
    #   validation data for early stopping and for ensemble construction.
    # - It is especially important to follow common best practices, such as _excluding_
    #   the additional split from fitting preprocessing parameters (e.g. feature
    #   statistics), and respecting dataset-specific splitting rules (e.g. time-based
    #   splitting). Otherwise, there is a high risk of overfitting.
    # - TabPack may struggle on small data where, due to limited split sizes, the
    #   additional split may not be representative enough for reliable early stopping
    #   and ensemble construction.
    #
    # After the first run, TabPack can be retrained on the union of the training and
    # validation data for better performance.
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=ratio
    )
    split = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    return {
        name: {k: v[idx] for k, v in dataset.items()} for name, idx in split.items()
    }


def preprocess_datasets(
    datasets: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Preprocess features."""
    datasets = {k: v.copy() for k, v in datasets.items()}

    # The following preprocessing of numerical features is a good alternative to usual
    # preprocessing strategies and is not specific to TabPack. That said, what works
    # best depends on a given dataset, as always.

    X_num_train = datasets['train']['x_num']
    num_transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(X_num_train) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    )
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, X_num_train.shape)
        .astype(X_num_train.dtype)
    )
    num_transformer.fit(
        X_num_train
        # The following perturbation deduplicates equal feature values and makes the
        # quantile transformer handle features with few distinct values more robustly.
        + noise
    )
    for dataset in datasets.values():
        dataset['x_num'] = num_transformer.transform(dataset['x_num'])

    return datasets


def _make_experiment_dir() -> Path:
    """Create an experiment directory for storing artifacts."""
    try:
        filename = __file__
    except NameError:  # Handle interactive notebooks.
        filename = Path.cwd() / 'tabpack'

    exp = Path(filename).with_suffix('') / datetime.datetime.now(datetime.UTC).strftime(
        '%Y%m%dT%H%M%S'
    )
    assert not exp.exists()
    exp.mkdir(parents=True)
    exp.joinpath('.gitignore').write_text('*')
    return exp


def main() -> None:
    # >>> Start

    exp = _make_experiment_dir()
    print(f'The experiment directory: {exp}')

    # >>> Reproducibility

    seeds = itertools.count()
    random.seed(next(seeds))
    np.random.seed(next(seeds))
    torch.manual_seed(next(seeds))

    # >>> Data

    # The data is prepared, split and preprocessed in a usual manner, nothing specific
    # to TabPack happens here.

    dataset, categorical_cardinalities = make_dataset()
    raw_datasets = split_dataset(dataset)
    datasets = preprocess_datasets(raw_datasets)
    del dataset

    n_classes = len(np.unique(np.concatenate([x['y'] for x in datasets.values()])))

    datasets = {
        name: {k: torch.as_tensor(v) for k, v in dataset.items()}
        for name, dataset in datasets.items()
    }

    # >>> Loss function

    # A loss function for a model pack must return a _one-dimensional_ tensor of the
    # shape `(pack_size,)` representing per-model losses.

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        assert input.ndim == 3  # Shape: (pack_size, batch_size, n_classes)
        assert target.ndim == 2  # Shape: (pack_size, batch_size)
        return (
            nn.functional.cross_entropy(
                # `cross_entropy` expects the second dimension to be the class
                # dimension, so we swap the batch and class dimensions.
                input.transpose(1, 2),
                target,
                reduction='none',  # Compute non-aggregated loss values.
            ).mean(1)  # Aggregate across the batch dimension.
        )  # Shape: (pack_size,)

    # >>> Metrics

    # A metric function for a model pack must return per-model metrics. The predictions
    # accepted by the metric functions are the model predictions transformed with
    # the `prediction_transform` function passed to `tabpackage.train`.

    def compute_accuracy_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tabpackage.metrics.accuracy_pack(
            y_true=y_true, y_pred=torch.argmax(y_pred, dim=-1)
        )

    def compute_cross_entropy_pack(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tabpackage.metrics.multiclass_cross_entropy_pack(
            y_true=y_true, y_proba=y_pred
        )

    # Traditional non-pack metrics can be obtained from pack metrics by reinterpreting
    # the input predictions as a pack of predictions with the pack size equal to one.

    def make_metric_fn_from_metric_pack_fn(metric_pack_fn):
        def metric_fn(*, y_true: Tensor, y_pred: Tensor) -> Tensor:
            return metric_pack_fn(y_true=y_true, y_pred=y_pred[None])[0]

        return metric_fn

    # Metric functions for the model pack.
    eval_metrics: dict[str, dict[str, tabpackage.MetricFn]] = {
        'val': {
            'accuracy': compute_accuracy_pack,
            'cross-entropy': compute_cross_entropy_pack,
        }
    }
    # The criterion guides the model checkpoint selection and early stopping.
    eval_criterion = tabpackage.EvalCriterionConfig(
        dataset='val',
        metric='accuracy',
        metric_direction=tabpackage.MetricDirection.MAXIMIZE,
    )
    # Metric functions for ensembles.
    ensemble_eval_metrics = {
        dataset_name: {
            k: make_metric_fn_from_metric_pack_fn(v)
            for k, v in dataset_eval_metrics.items()
        }
        for dataset_name, dataset_eval_metrics in eval_metrics.items()
    }

    # >>> Per-model hyperparameters

    # Different models are obtained by sampling hyperparameters randomly. Note that
    # individual models are not expected to be powerful: they only should provide
    # meaningful pool of candidates for building powerful ensembles.

    rng = random.Random(next(seeds))

    def random_loguniform(a: float, b: float) -> float:
        return math.exp(rng.uniform(math.log(a), math.log(b)))

    # Sample per-model hyperparameters randomly for all models in advance. The
    # distribution used below is the one used in the TabPack paper.
    per_model_hyperparameters = [
        {
            'model': {
                'numerical_embeddings': {
                    'd_embedding': rng.randint(8, 32),
                    'init_scale': random_loguniform(0.01, 10.0),
                },
                'n_blocks': rng.randint(1, 3),
                'dropout': rng.uniform(0.0, 0.5) if rng.randint(0, 1) else 0.0,
            },
            'optimizer': {
                'adamw_lr': random_loguniform(0.0001, 0.005),
                'weight_decay': random_loguniform(0.001, 1.0),
                'muon_lr': random_loguniform(0.001, 0.1),
            },
        }
        for _ in range(64)
    ]

    # >>> Efficiency

    # Optimal efficiency-related settings vary significantly depending on the workload
    # and hardware. What works well for a small model pack trained on small data on a
    # MacBook may not be the best fit for a large model packs trained on big data on an
    # industry-grade GPU. The package comes with all efficiency tweaks _disabled_ by
    # default, with guidelines on how to use them in the comments in this script.
    #
    # One of possible approaches to optimizing training efficiency is as follows (each
    # of the points below corresponds to some options in `tabpackage.train`):
    #
    # - Try enabling autocast.
    # - Try enabling compilation.
    # - Try the indexed implementation of the forward pass with compilation
    #   disabled/enabled.

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.mps.is_available()
        else 'cpu'
    )

    # >>> Training

    print('\n>>> Training\n')
    tabpack = tabpackage.train(
        datasets={
            # Move the datasets to the main device for faster runtimes.
            dataset_name: {k: v.to(device) for k, v in dataset.items()}
            for dataset_name, dataset in datasets.items()
        },
        model={
            # Cardinalities of categorical features cannot be inferred reliably from
            # training data and must be provided explicitly.
            'categorical_cardinalities': categorical_cardinalities,
            'd_out': n_classes,
            'numerical_embeddings': {'type': tabpackage.nn.CosineEmbeddingsPack},
            'd_block': 384,
            # For heavy workloads, the "indexed" implementation of the forward pass may
            # be faster than the "dense" one (the default).
            # 'forward_impl': 'indexed'
        },
        # A task-specific transformation of model predictions. Evaluation metrics are
        # computed for predictions transformed with this function. To make the resulting
        # TabPack exportable with `torch.save`, use transformations that can be pickled
        # and imported.
        prediction_transform=partial(torch.softmax, dim=-1),
        loss_fn=loss_fn,
        optimizer={
            'type': tabpackage.optim.MuonAdamWPack,
            # For some combinations of hardware and workload, excluding unused blocks
            # from some of the Muon computations can accelerate training.
            'muon_skip_zero_updates': (
                # Unused blocks can occur only in depth-varying models.
                'n_blocks' in per_model_hyperparameters[0].get('model', {})
                # The following condition is a heuristic that may or may not be optimal
                # depending on a dataset.
                and device.type == 'mps'
            ),
        },
        batch_size=256,
        max_epochs=None,
        early_stopping_patience=16,
        per_model_hyperparameters=per_model_hyperparameters,
        eval_metrics=eval_metrics,
        eval_criterion=eval_criterion,
        eval_batch_size=32768,
        ensembles={
            'greedy': tabpackage.EnsembleConfig(
                algorithm={
                    'type': tabpackage.ensemble.greedy_ensemble,
                    'max_ensemble_size': 32,
                },
                online=True,
                # Aggregating probabilities produced by ensemble members can round
                # slightly above 1 because of limitations of floating-point arithmetics.
                # Also, the resulting probabilities may not sum to 1.0. Provide the
                # corresponding prediction transformation required for a given task.
                prediction_transform=partial(torch.clamp_max, max=1.0),
                early_stopping_patience=32,
                algorithm_metric=compute_accuracy_pack,
                algorithm_metric_direction=tabpackage.MetricDirection.MAXIMIZE,
                # Cross-entropy is worth trying for ensemble selection when the main
                # task metric is "discrete" (like accuracy). The effect can be both
                # positive and negative.
                # algorithm_metric=compute_cross_entropy_pack,
                # algorithm_metric_direction=tabpackage.MetricDirection.MINIMIZE,
                algorithm_dataset='val',
                # In some cases, setting the algorithm device to `aux_device` results in
                # faster runtimes.
                algorithm_device=device,
            ),
            'top-k': tabpackage.EnsembleConfig(
                algorithm={'type': tabpackage.ensemble.topk_ensemble, 'k': 8},
                online=False,
                prediction_transform=partial(torch.clamp_max, max=1.0),
                early_stopping_patience=None,
                algorithm_metric=compute_accuracy_pack,
                algorithm_metric_direction=tabpackage.MetricDirection.MAXIMIZE,
                algorithm_dataset='val',
                algorithm_device=device,
            ),
            # ... more ensembles can be added.
        },
        ensemble_eval_metrics=ensemble_eval_metrics,
        ensemble_eval_criterion=eval_criterion,
        stop_when_all_ensembles_are_finalized=True,
        seed=next(seeds),
        # If training all models at once leads to OOM, consider limiting the number of
        # models trained simultanesously. This affects online ensemble construction, so,
        # similarly to reducing model dimensions and/or batch size, using the following
        # option may come with its trade-offs.
        # max_pack_size=...,
        device=device,
        aux_device=torch.device(
            # On machines with MPS devices, GPUs share system memory with the CPU.
            # Keeping auxiliary tensors on MPS avoids unnecessary copies between the CPU
            # and the device.
            'mps' if device.type == 'mps' else 'cpu'
        ),
        # Using autocast makes heavy training workloads faster, while lightweight
        # workloads can become _slower_, depending on hardware. Adjust the option
        # accordingly.
        autocast_dtype=(
            torch.bfloat16
            if (
                (device.type == 'cuda' and torch.cuda.is_bf16_supported())
                or device.type == 'mps'
            )
            else None
        ),
        # Compilation is an experimental option. It is recommended to validate results
        # against an uncompiled baseline in the target environment, as behavior can vary
        # with hardware and software versions. In many cases, compilation gives
        # meaningful speed-ups. Compiling the full model may also change which TabPack
        # forward-pass implementation is faster: dense or indexed.
        #
        # The full model compilation is supported only for CUDA.
        # compile_model=device.type == 'cuda',
        # compile_model_numerical_embeddings=device.type == 'mps',
        # compile_model_first_block=device.type == 'mps',
        # compile_optimizer_muon_newton_schulz=True,
    )

    # >>> (Optional, but recommended) Retraining on full data for better performance

    # Below, members of one of TabPack's ensembles are retrained from scratch on full
    # training data, i.e. _without_ extracting any sort of validation splits from it.
    # This is rather a heuristic attempt to improve the ensemble's performance than an
    # established method. That said, from preliminary benchmarks, it seems to be
    # beneficial on some datasets, and therefore is recommended for testing in practice.

    retrain = True
    if retrain:
        # Merge the training and validation data.
        retraining_raw_datasets = {
            'train': {
                k: np.concatenate([v, raw_datasets['val'][k]])
                for k, v in raw_datasets['train'].items()
            },
            # Include the test data as well so that it undergoes the new preprocessing.
            'test': raw_datasets['test'],
        }
        # Redo the preprocessing on the full training data.
        retraining_datasets = preprocess_datasets(retraining_raw_datasets)
        retraining_datasets = {
            name: {k: torch.as_tensor(v, device=device) for k, v in dataset.items()}
            for name, dataset in retraining_datasets.items()
        }

        print('\n>>> Retraining\n')
        retrained_tabpack = tabpackage.retrain(
            tabpack,
            ensemble_name='greedy',
            dataset=retraining_datasets['train'],
            # Optionally, increase the ensemble size by repeating each ensemble member
            # multiple times, which may give a performance boost in some cases.
            # n_repeats=2,
            seed=next(seeds),
        )
        datasets = retraining_datasets

    else:
        retrained_tabpack = None

    # >>> Making predictions

    # `predictor` is a plain PyTorch module representing one or more ensembles, powered
    # by a model pack storing a union of model checkpoints participating in the
    # ensembles.
    predictor = (
        # For demonstration purposes, all ensembles are included in the predictor.
        tabpack.make_predictor(ensemble_names=tabpack.get_ensemble_names())
        if retrained_tabpack is None
        # In the retrained TabPack, only one ensemble is available.
        else retrained_tabpack.make_predictor()
    )
    predictor.to(device)

    dataset = {k: v.to(device) for k, v in datasets['test'].items()}
    dataset_size = len(dataset['y'])
    batch_size = 4096

    # For heavy workloads with large batch sizes, the indexed forward pass may be more
    # efficient.
    # predictor.model_pack.backbone.set_forward_impl('indexed')

    # Per-model predictions. Shape: [(n_ensemble_members, batch_size, n_classes), ...]
    # (`n_ensemble_members` is the number of model checkpoints participating in at least
    # one of the specified ensembles; if a checkpoint is a member of more than one
    # ensemble, it is counted only once).
    y_pred_pack_batches: list[Tensor] = []

    # Ensemble predictions. Shape: {ensemble_name: [(batch_size, n_classes), ...], ...}.
    y_pred_ensembles_batches: dict[str, list[Tensor]] = {
        name: [] for name in predictor.ensembles
    }

    with torch.inference_mode():
        for i in range(0, dataset_size, batch_size):
            y_pred_pack, y_pred_ensembles = predictor(
                dataset['x_num'][i : i + batch_size] if 'x_num' in dataset else None,
                dataset['x_cat'][i : i + batch_size] if 'x_cat' in dataset else None,
            )
            y_pred_pack_batches.append(y_pred_pack)
            for ensemble_name, y_pred in y_pred_ensembles.items():
                y_pred_ensembles_batches[ensemble_name].append(y_pred)

    y_pred_pack = torch.cat(y_pred_pack_batches, dim=1)
    assert y_pred_pack.shape == (
        predictor.model_pack.pack_size,
        dataset_size,
        n_classes,
    )

    for ensemble_name, ensemble_y_pred_batches in y_pred_ensembles_batches.items():
        ensemble_y_pred = torch.cat(ensemble_y_pred_batches)
        assert ensemble_y_pred.shape == (dataset_size, n_classes)
        del ensemble_name

    # >>> Export setup

    predictor.cpu()

    example_batch_size = 1
    example_batch = {
        x: dataset[x][:example_batch_size].cpu()
        for x in ('x_num', 'x_cat')
        if x in dataset
    }
    example_batch_as_tuple = (example_batch.get('x_num'), example_batch.get('x_cat'))

    # >>> Export with `torch.save`

    # `torch.save` uses Pickle, so this export strategy requires the prediction
    # transformations passed to `tabpackage.train` to be importable.
    torch.save(predictor, exp / 'predictor.pt')

    # >>> Export with ONNX

    # Export requirements: `onnx>=1.17.0`, `onnxscript>=0.6.1`.
    # Deployment requirements: `onnxruntime>=1.23.2`.

    # The "dense" forward pass must be used for the ONNX export.
    predictor.model_pack.backbone.set_forward_impl('dense')
    torch.onnx.export(
        predictor,
        example_batch_as_tuple,
        exp / 'predictor.onnx',
        dynamo=True,
        opset_version=18,
        external_data=False,
        input_names=list(example_batch),
        output_names=['pack', *(f'ensemble_{name}' for name in predictor.ensembles)],
    )

    # >>> Export with `torch.export`

    # Deploying a model exported with `torch.export` using AOTInductor and ExecuTorch
    # requires rigorous testing for the target environment, as different software
    # versions may require different workarounds. A (most likely, incomplete) list of
    # examples:
    #
    # - When deploying with AOTInductor with PyTorch 2.10 and older, the
    #   `from torch._inductor import codecache` import must be added when loading the
    #   model in a fresh process.
    # - When compiling the exported module for AOTInductor in the same process where the
    #   module was created, for some versions of PyTorch, the exported module must be
    #   reloaded from the export file before the compilation.
    # - When using XNNPACK with ExecuTorch<1.2, `XnnpackPartitioner(per_op_mode=True)`
    #   must be used; otherwise, results can be silently incorrect.

    # The "dense" forward pass must be used for exporting TabPack with `torch.export`.
    predictor.model_pack.backbone.set_forward_impl('dense')
    exported = torch.export.export(
        predictor,
        example_batch_as_tuple,
        strict=torch.torch_version.TorchVersion(torch.__version__) >= '2.8',
    )
    torch.export.save(exported, exp / 'predictor.pt2')


if __name__ == '__main__':
    main()
