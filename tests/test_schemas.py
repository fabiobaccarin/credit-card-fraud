from functools import partial
from typing import Any

import pytest
from hypothesis import given
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from credit_card_fraud import schemas as sch

from . import strategies as cst


def get_nested(data: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        data = data.get(key)  # type: ignore
        if data is None:
            return data

    return data


def get_defaults(model_class: type[sch.Schema]) -> dict[str, Any]:
    defaults = dict()
    for name, field_info in model_class.model_fields.items():
        has_default_value = field_info.default is not PydanticUndefined
        if has_default_value:
            defaults[name] = field_info.default

        has_default_factory = field_info.default_factory is not None
        if has_default_factory:
            default = field_info.default_factory()  # type: ignore
            if issubclass(type(default), BaseModel):
                default = get_defaults(type(default))
                defaults[name] = default
            else:
                defaults[name] = default

    return defaults


class TestConfig:
    @staticmethod
    def get_values(data: dict[str, Any]) -> dict[str, Any]:
        gn = partial(get_nested, data)

        return dict(
            impute_strategy=gn("preprocessing.impute_strategy"),
            scaling_method=gn("preprocessing.scaling_method"),
            outlier_remove=gn("preprocessing.outlier.remove"),
            outlier_method=gn("preprocessing.outlier.method"),
            max_features=gn("features.max_features"),
            selection_method=gn("features.selection_method"),
            correlation_threshold=gn("features.correlation_threshold"),
            name=gn("model.name"),
            model_type=gn("model.model_type"),
            random_state=gn("model.random_state"),
        )

    @staticmethod
    def assert_config_values_equal(config: sch.Config, values: dict[str, Any | None]) -> None:
        # Preprocessing
        assert config.preprocessing.impute_strategy == values.get("impute_strategy")
        assert config.preprocessing.scaling_method == values.get("scaling_method")
        assert config.preprocessing.outlier.remove == values.get("outlier_remove")
        assert config.preprocessing.outlier.method == values.get("outlier_method")

        # Features
        assert config.features.max_features == values.get("max_features")
        assert config.features.selection_method == values.get("selection_method")
        assert config.features.correlation_threshold == values.get("correlation_threshold")

        # Model
        assert config.model.name == values.get("name")
        assert config.model.model_type == values.get("model_type")
        assert config.model.random_state == values.get("random_state")

    @pytest.mark.property
    @given(cst.full_config_strategy())
    def test_initializes_correctly(self, config: sch.Config) -> None:
        assert config is not None

    @pytest.mark.property
    @given(cst.required_config_strategy())
    def test_initializes_default_values_correctly(self, config: sch.Config) -> None:
        assert config is not None

    @pytest.mark.unit
    def test_initialization_without_name_raises_exception(self) -> None:
        with pytest.raises(ValidationError):
            sch.Config(**{})
