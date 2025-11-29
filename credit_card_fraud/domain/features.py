"""Pydantic models related to feature handling"""

from typing import Annotated

from pydantic import Field

from . import base, types


class FeatureSelection(base.FrozenModel):
    selected_features: Annotated[
        types.FeatureList, Field(description="List of selected features names")
    ]

    dropped_features: Annotated[
        types.FeatureList, Field(description="List of dropped features names")
    ]
