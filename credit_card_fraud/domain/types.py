"""Custom types"""

from enum import StrEnum


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
