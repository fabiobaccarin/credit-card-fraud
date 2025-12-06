from functools import partial
from typing import Any

import pytest
from hypothesis import given

from credit_card_fraud import schemas as sch

from . import strategies as cst


def get_nested(data: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        data = data.get(key, dict())

    return data


@pytest.mark.property
@given(cst.config_strategy())
def test_config_schema_initializes_correctly(data: dict[str, Any]) -> None:
    config = sch.Config(**data)
    gn = partial(get_nested, data)

    # Preprocessing
    impute_strategy = gn("preprocessing.impute_strategy")
    scaling_method = gn("preprocessing.scaling_method")
    outlier_remove = gn("preprocessing.outlier.remove")
    outlier_method = gn("preprocessing.outlier.method")

    # Features
    max_features = gn("features.max_features")
    selection_method = gn("features.selection_method")
    correlation_threshold = gn("features.correlation_threshold")

    # Model
    name = gn("model.name")
    model_type = gn("model.model_type")
    random_state = gn("model.random_state")

    assert config is not None

    # Preprocessing
    assert config.preprocessing.impute_strategy == impute_strategy
    assert config.preprocessing.scaling_method == scaling_method
    assert config.preprocessing.outlier.remove == outlier_remove
    assert config.preprocessing.outlier.method == outlier_method

    # Features
    assert config.features.max_features == max_features
    assert config.features.selection_method == selection_method
    assert config.features.correlation_threshold == correlation_threshold

    # Model
    assert config.model.name == name
    assert config.model.model_type == model_type
    assert config.model.random_state == random_state
