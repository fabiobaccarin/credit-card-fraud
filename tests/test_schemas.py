import pytest
from hypothesis import given
from pydantic import ValidationError

from credit_card_fraud import schemas as sch

from . import strategies as cst


class TestConfig:
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
