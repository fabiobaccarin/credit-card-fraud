"""Pydantic configuration models"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StrictBool

from . import types

__ALL__: list[str] = ["PreprocessingConfig"]


class _BaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, validate_assignment=True, arbitrary_types_allowed=True)


class _OutlierConfig(_BaseConfig):
    remove: Annotated[
        StrictBool, Field(default=False, description="Whether to remove outliers from the dataset")
    ]

    method: Annotated[
        types.OutlierFindMethod,
        Field(
            default=types.OutlierFindMethod.IQR,
            description="Method used to find outliers for removal. Only applied if removed is True",
        ),
    ]


class PreprocessingConfig(_BaseConfig):
    impute_strategy: Annotated[
        types.ImputeStrategy,
        Field(
            default=types.ImputeStrategy.MEAN,
            description="Strategy to use for imputing missing values",
        ),
    ]

    scaling_method: Annotated[
        types.ScalingMethod,
        Field(
            default=types.ScalingMethod.STANDARD,
            description="Scaling method to use for feature scaling",
        ),
    ]

    outlier: _OutlierConfig


class FeatureConfig(_BaseConfig):
    max_features: Annotated[
        PositiveInt, Field(default=5, description="Maximum number of features to select")
    ]

    selection_method: Annotated[
        types.FeatureSelectionMethod,
        Field(
            default=types.FeatureSelectionMethod.F_CLASSIF,
            description="Which method to use for feature selection",
        ),
    ]
