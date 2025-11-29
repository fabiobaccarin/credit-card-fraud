"""Project constants"""

from typing import Final, NamedTuple

OPENML_DATASET_ID: Final = 1597
TARGET: Final = "Class"


class MLFlow(NamedTuple):
    EXPERIMENT_NAME: str = "Credit card fraud"
    TRACKING_URI: str = "http://localhost:5000"
