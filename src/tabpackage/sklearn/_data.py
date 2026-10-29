import dataclasses
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch import Tensor

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
else:
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        if error.name != 'pandas':
            raise
        pd = None

    try:
        import polars as pl
    except ModuleNotFoundError as error:
        if error.name != 'polars':
            raise
        pl = None

XLike: TypeAlias = 'np.ndarray | pd.DataFrame | pl.DataFrame'
YLike: TypeAlias = 'np.ndarray | pd.Series | pl.Series'
YPredLike: TypeAlias = np.ndarray
DType: TypeAlias = 'np.dtype | pd.api.extensions.ExtensionDtype | pl.DataType'


def _get_dataframe_values(
    X: 'pd.DataFrame | pl.DataFrame', columns: list[str] | tuple[str, ...]
) -> np.ndarray:
    return (
        X[list(columns)].to_numpy(writable=True)
        if pl is not None and isinstance(X, pl.DataFrame)
        else X[list(columns)].to_numpy()
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FeatureSchema:
    container: type
    dtypes: tuple[DType, ...]
    columns: tuple[str, ...] | None

    @property
    def n_features(self) -> int:
        return len(self.dtypes)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, dtype in zip(self.columns or (), self.dtypes)
            if not (
                (isinstance(dtype, np.dtype) and dtype == np.float32)
                or (pl is not None and dtype == pl.Float32)
            )
        )

    @property
    def n_numerical_features(self) -> int:
        return self.n_features - len(self.categorical_columns)

    @classmethod
    def from_data(cls, X: XLike) -> 'FeatureSchema':
        if np.ma.isMaskedArray(X):
            raise TypeError('Masked arrays are not supported')

        if len(X.shape) != 2:
            raise ValueError('`X` must be two-dimensional')

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError('`X` must have at least one row and one column')

        if pd is not None and isinstance(X, pd.DataFrame):
            if not X.columns.is_unique:
                raise ValueError('Feature column names must be unique')

            if not all(isinstance(name, str) for name in X.columns):
                raise TypeError('Feature column names must be strings')

            columns = tuple(X.columns)
            dtypes = tuple(X.dtypes)
            for name, dtype in zip(columns, dtypes, strict=True):
                column = X[name]
                if isinstance(dtype, np.dtype) and dtype == np.float32:
                    if not np.isfinite(column.to_numpy()).all():
                        raise ValueError(
                            'Numerical features must contain only finite values'
                        )
                else:
                    if isinstance(dtype, pd.CategoricalDtype):
                        categories = dtype.categories
                        if not (
                            _is_string_dtype(categories.dtype)
                            or (
                                categories.dtype == object
                                and pd.api.types.infer_dtype(categories, skipna=False)
                                == 'string'
                            )
                        ):
                            raise TypeError('Feature categories must be strings')
                    elif not _is_string_dtype(dtype):
                        raise TypeError(
                            'Numerical features require `numpy.float32`; categorical'
                            ' features require a string data type or string categories'
                        )

                    if column.isna().any():
                        raise ValueError(
                            'Categorical features must not contain missing values'
                        )
        elif pl is not None and isinstance(X, pl.DataFrame):
            columns = tuple(X.columns)
            dtypes = tuple(X.dtypes)
            for name, dtype in zip(columns, dtypes, strict=True):
                column = X[name]
                if column.null_count() != 0:
                    raise ValueError('Features must not contain missing values')

                if dtype == pl.Float32:
                    if not column.is_finite().all():
                        raise ValueError(
                            'Numerical features must contain only finite values'
                        )
                elif not (
                    dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))
                ):
                    raise TypeError(
                        'Polars feature columns require `polars.Float32`,'
                        ' `polars.String`, `polars.Categorical`, or `polars.Enum`'
                    )
        else:
            columns = None
            dtypes = (X.dtype,) * X.shape[1]
            if X.dtype != np.float32:
                raise TypeError('`X` must have the data type `numpy.float32`')

            if not np.isfinite(X).all():
                raise ValueError('`X` must contain only finite values')

        return cls(container=type(X), dtypes=dtypes, columns=columns)

    def make_encoder(self, X: XLike) -> OrdinalEncoder | None:
        return (
            None
            if isinstance(X, np.ndarray) or not self.categorical_columns
            else OrdinalEncoder(
                dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1
            ).fit(_get_dataframe_values(X, self.categorical_columns))
        )

    def encode(self, X: XLike, encoder: OrdinalEncoder | None) -> dict[str, np.ndarray]:
        if isinstance(X, np.ndarray):
            return {'x_num': X}

        categorical_columns = self.categorical_columns
        numerical_columns = [
            name for name in X.columns if name not in categorical_columns
        ]

        arrays = {}
        if numerical_columns:
            arrays['x_num'] = _get_dataframe_values(X, numerical_columns)
        if encoder is not None:
            codes = encoder.transform(_get_dataframe_values(X, categorical_columns))
            cardinalities = np.array([len(values) for values in encoder.categories_])
            arrays['x_cat'] = np.where(codes == -1, cardinalities, codes)

        return arrays


def _is_numpy_string_dtype(dtype: np.dtype) -> bool:
    string_dtype = getattr(getattr(np, 'dtypes', None), 'StringDType', None)
    return dtype.kind == 'U' or (
        string_dtype is not None
        and isinstance(dtype, string_dtype)
        and not hasattr(dtype, 'na_object')
    )


def _is_string_dtype(dtype: DType) -> bool:
    return (pd is not None and isinstance(dtype, pd.StringDtype)) or (
        isinstance(dtype, np.dtype) and _is_numpy_string_dtype(dtype)
    )


def _get_label_values(y: YLike) -> np.ndarray:
    return (
        y.to_numpy(writable=True)
        if pl is not None and isinstance(y, pl.Series)
        else (
            y.to_numpy()
            if pd is not None and isinstance(y, pd.Series)
            else (y.astype(object) if y.dtype.kind == 'T' else y)
        )
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabelSchema:
    container: type
    dtype: DType
    classes: np.ndarray | None

    @classmethod
    def from_data(cls, y: YLike, *, classification: bool) -> 'LabelSchema':
        if np.ma.isMaskedArray(y):
            raise TypeError('Masked arrays are not supported')

        if len(y.shape) != 1:
            raise ValueError('`y` must be one-dimensional')

        if pl is not None and isinstance(y, pl.Series):
            if y.dtype not in (
                (pl.Int64, pl.String) if classification else (pl.Float32,)
            ):
                raise TypeError(
                    'Classification targets require `polars.Int64` or `polars.String`'
                    if classification
                    else 'Regression targets require `polars.Float32`'
                )

            if y.null_count() != 0:
                raise ValueError('`y` must not contain missing values')
        else:
            numeric_dtype = np.int64 if classification else np.float32
            if not (
                (isinstance(y.dtype, np.dtype) and y.dtype == numeric_dtype)
                or (classification and _is_string_dtype(y.dtype))
            ):
                raise TypeError(
                    'Classification targets require `numpy.int64` or a string data type'
                    if classification
                    else 'Regression targets require `numpy.float32`'
                )

            if pd is not None and isinstance(y, pd.Series) and y.isna().any():
                raise ValueError('`y` must not contain missing values')

        values = _get_label_values(y)
        if classification:
            classes = np.unique(values)
            if len(classes) < 2:
                raise ValueError(
                    'Classification targets must contain at least two classes'
                )
        else:
            if not np.isfinite(values).all():
                raise ValueError('Regression targets must contain only finite values')

            classes = None

        return cls(container=type(y), dtype=y.dtype, classes=classes)

    def validate(self, y: YLike) -> None:
        if np.ma.isMaskedArray(y):
            raise TypeError('Masked arrays are not supported')

        if type(y) is not self.container:
            raise TypeError(
                'Labels must use the same container type as the training labels'
            )

        if len(y.shape) != 1:
            raise ValueError('`y` must be one-dimensional')

        if y.dtype != self.dtype:
            raise TypeError(
                'Labels must have the same data type as the training labels'
            )

        if pl is not None and isinstance(y, pl.Series):
            if y.null_count() != 0:
                raise ValueError('`y` must not contain missing values')
        elif pd is not None and isinstance(y, pd.Series) and y.isna().any():
            raise ValueError('`y` must not contain missing values')

        values = _get_label_values(y)
        if self.classes is None:
            if not np.isfinite(values).all():
                raise ValueError('Regression targets must contain only finite values')
        elif not np.isin(values, self.classes).all():
            raise ValueError(
                'Labels contain classes that were not present in the training data'
            )

    def encode(self, y: YLike) -> np.ndarray:
        values = _get_label_values(y)
        return (
            values
            if self.classes is None
            else np.searchsorted(self.classes, values).astype(np.int64)
        )


def make_dataset(
    arrays: dict[str, np.ndarray], *, device: torch.device
) -> dict[str, Tensor]:
    return {
        name: torch.as_tensor(np.ascontiguousarray(array), device=device)
        for name, array in arrays.items()
    }
