"""Custom types"""

from abc import abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

from pydantic import Field, StrictStr

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray


# ==================================================================================================
# Basic Types
# ==================================================================================================
type Matrix = pd.DataFrame | np.ndarray
type TargetVector = pd.Series[np.float64] | NDArray[np.float64]
type FeatureList = Annotated[list[StrictStr], Field(default_factory=list, min_length=1)]


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
    def fit(self, X: Matrix, y: TargetVector) -> Any: ...

    @abstractmethod
    def predict(self, X: Matrix) -> np.ndarray: ...


@runtime_checkable
class ProbabilisticClassifier(_BaseClassifier, Protocol):
    @abstractmethod
    def predict_proba(self, X: Matrix) -> np.ndarray: ...


@runtime_checkable
class NonProbabilisticClassifier(_BaseClassifier, Protocol):
    @abstractmethod
    def decision_function(self, X: Matrix) -> np.ndarray: ...
