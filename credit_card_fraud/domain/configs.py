"""Pydantic configuration models"""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    StrictStr,
)

from . import types

__ALL__: list[str] = ["Config"]


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


class _PreprocessingConfig(_BaseConfig):
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


class _FeatureConfig(_BaseConfig):
    max_features: Annotated[
        PositiveInt | None, Field(default=5, description="Maximum number of features to select")
    ]

    selection_method: Annotated[
        types.FeatureSelectionMethod,
        Field(
            default=types.FeatureSelectionMethod.F_CLASSIF,
            description="Which method to use for feature selection",
        ),
    ]

    correlation_threshold: Annotated[
        PositiveFloat,
        Field(
            default=0.75,
            le=1.0,
            description="Correlation threshold above which features should be considered redundant",
        ),
    ]


class _ModelConfig(_BaseConfig):
    name: Annotated[
        StrictStr,
        Field(
            default=...,
            min_length=1,
            description="Model name used for reference (used as run name for MLFlow logging)",
        ),
    ]

    model_type: Annotated[
        types.ModelType, Field(default=types.ModelType.SKLEARN, description="Model type to use")
    ]

    random_state: Annotated[
        PositiveInt,
        Field(default=0, description="Random state to seed randomness and ensure reproducibility"),
    ]


class Config(_BaseConfig):
    preprocessing: _PreprocessingConfig = Field(default_factory=lambda: _PreprocessingConfig())  # type: ignore
    features: _FeatureConfig = Field(default_factory=lambda: _FeatureConfig())  # type: ignore
    model: _ModelConfig = Field(default_factory=lambda: _ModelConfig())  # type: ignore
