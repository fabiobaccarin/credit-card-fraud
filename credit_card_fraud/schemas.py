from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StrictBool,
)

from . import types as t

__all__ = ["FrozenModel", "Config", "FeatureSelection", "Schema"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, validate_assignment=True, arbitrary_types_allowed=True, use_enum_values=True
    )


class _OutlierConfig(FrozenModel):
    remove: Annotated[
        StrictBool, Field(default=False, description="Whether to remove outliers from the dataset")
    ]

    method: Annotated[
        t.OutlierFindMethod,
        Field(
            default=t.OutlierFindMethod.IQR,
            description="Method used to find outliers for removal. Only applied if removed is True",
        ),
    ]


class _PreprocessingConfig(FrozenModel):
    impute_strategy: Annotated[
        t.ImputeStrategy,
        Field(
            default=t.ImputeStrategy.MEAN,
            description="Strategy to use for imputing missing values",
        ),
    ]

    scaling_method: Annotated[
        t.ScalingMethod,
        Field(
            default=t.ScalingMethod.STANDARD,
            description="Scaling method to use for feature scaling",
        ),
    ]

    outlier: Annotated[_OutlierConfig, Field(default_factory=lambda: _OutlierConfig(**{}))]


class _FeatureConfig(FrozenModel):
    max_features: Annotated[
        PositiveInt | None, Field(default=5, description="Maximum number of features to select")
    ]

    selection_method: Annotated[
        t.FeatureSelectionMethod,
        Field(
            default=t.FeatureSelectionMethod.F_CLASSIF,
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


class _ModelConfig(FrozenModel):
    name: Annotated[
        t.NonEmptyStr,
        Field(
            default=...,
            description="Model name used for reference (used as run name for MLFlow logging)",
        ),
    ]

    model_type: Annotated[
        t.ModelType, Field(default=t.ModelType.SKLEARN, description="Model type to use")
    ]

    random_state: Annotated[
        NonNegativeInt,
        Field(default=0, description="Random state to seed randomness and ensure reproducibility"),
    ]


class Config(FrozenModel):
    preprocessing: _PreprocessingConfig = Field(
        default_factory=lambda: _PreprocessingConfig(**dict())
    )
    features: _FeatureConfig = Field(default_factory=lambda: _FeatureConfig(**dict()))
    model: _ModelConfig = Field(default_factory=lambda: _ModelConfig(**dict()))


class FeatureSelection(FrozenModel):
    selected_features: Annotated[
        t.FeatureList,
        Field(default_factory=lambda: list(), description="List of selected features names"),
    ]

    dropped_features: Annotated[
        t.FeatureList,
        Field(default_factory=lambda: list(), description="List of dropped features names"),
    ]


type Schema = Config | FeatureSelection
