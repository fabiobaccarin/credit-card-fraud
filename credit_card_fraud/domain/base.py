"""Base models to share Pydantic model configuration across different models"""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, validate_assignment=True, arbitrary_types_allowed=True)
