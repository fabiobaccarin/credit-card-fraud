from typing import Any, Callable, Final, cast

import pytest

from credit_card_fraud import schemas as sch

_SCHEMAS: Final[list[type[sch.Schema]]] = [sch.Config]


@pytest.fixture(params=_SCHEMAS)
def schema_factory(request: pytest.FixtureRequest) -> Callable[[dict[str, Any]], sch.Schema]:
    schema_class = cast(type[sch.Schema], request.param)

    def _factory(data: dict[str, Any]) -> sch.Schema:
        return schema_class(**data)

    return _factory
