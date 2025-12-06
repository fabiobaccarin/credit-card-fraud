"""Custom types"""

from abc import abstractmethod
from enum import StrEnum
from typing import Annotated, Any, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field, StrictStr

# ==================================================================================================
# Basic Types
# ==================================================================================================
type NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
type Matrix = pd.DataFrame | NDArray
type TargetVector = pd.Series[np.float64] | np.ndarray[tuple[int], np.dtype[np.float64]]
type FeatureList = list[NonEmptyStr]


# ==================================================================================================
# Enum Types
# ==================================================================================================
class ScalingMethod(StrEnum):
    STANDARD = "standard"
    MINMAX = "minmax"
    MAXABS = "maxabs"


class ImputeStrategy(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    CONSTANT = "constant"


class OutlierFindMethod(StrEnum):
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


class FeatureSelectionMethod(StrEnum):
    LASSO = "lasso"
    F_CLASSIF = "f_classif"
    MUTUAL_INFO = "mutual_info"
    SEQUENTIAL_FORWARD = "sequential_forward"
    SEQUENTIAL_BACKWARD = "sequential_backward"
    RFE = "rfe"
    RFE_CV = "rfe_cv"


class ModelType(StrEnum):
    SKLEARN = "sklearn"
    XGBOOT = "xgboost"
    LIGHTGBM = "lightgbm"


# ==================================================================================================
# Protocols
# ==================================================================================================
class _BaseClassifier(Protocol):
    @abstractmethod
    def fit(self, X: Any, y: Any) -> Self: ...

    @abstractmethod
    def predict(self, X: Any) -> Any: ...


@runtime_checkable
class ProbabilisticClassifier(_BaseClassifier, Protocol):
    @abstractmethod
    def predict_proba(self, X: Any) -> Any: ...


@runtime_checkable
class NonProbabilisticClassifier(_BaseClassifier, Protocol):
    @abstractmethod
    def decision_function(self, X: Any) -> Any: ...


# ==================================================================================================
# Union Types
# ==================================================================================================
type Classifier = ProbabilisticClassifier | NonProbabilisticClassifier
