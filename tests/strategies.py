import string
from typing import Any, Callable, Final

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from credit_card_fraud import constants, types

ASCII_LOWERCASE_WITH_DIGITS: Final = string.ascii_lowercase + string.digits

settings.register_profile(
    name="default",
    parent=settings.get_profile("default"),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,
)


@st.composite
def config_strategy(draw: Callable) -> dict[str, Any]:
    # Preprocessing
    impute_strategy = draw(st.sampled_from(list(types.ImputeStrategy)))
    scaling_method = draw(st.sampled_from(list(types.ScalingMethod)))
    outlier_remove = draw(st.booleans())
    outlier_method = draw(st.sampled_from(list(types.OutlierFindMethod)))

    # Features
    max_features = draw(st.integers(min_value=1, max_value=20))
    selection_method = draw(st.sampled_from(list(types.FeatureSelectionMethod)))
    correlation_threshold = draw(st.floats(min_value=0.7, max_value=0.9))

    # Model
    name = draw(st.text(alphabet=ASCII_LOWERCASE_WITH_DIGITS, min_size=5, max_size=20))
    model_type = draw(st.sampled_from(list(types.ModelType)))
    random_state = draw(st.integers(min_value=0, max_value=constants.MAX_RANDOM_STATE))

    return dict(
        preprocessing=dict(
            impute_strategy=impute_strategy,
            scaling_method=scaling_method,
            outlier=dict(remove=outlier_remove, method=outlier_method),
        ),
        features=dict(
            max_features=max_features,
            selection_method=selection_method,
            correlation_threshold=correlation_threshold,
        ),
        model=dict(name=name, model_type=model_type, random_state=random_state),
    )
