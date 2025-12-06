"""Project constants"""

from typing import Final, Literal

OPENML_DATASET_ID: Literal[1597] = 1597
TARGET: Literal["Class"] = "Class"

MLFLOW_EXPERIMENT_NAME: str = "Credit card fraud"
MLFLOW_TRACKING_URI: str = "http://localhost:5000"


MAX_RANDOM_STATE: Final = 2**31 - 1
