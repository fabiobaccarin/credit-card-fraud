import string
from typing import Final

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from credit_card_fraud import constants as c
from credit_card_fraud import schemas as sch
from credit_card_fraud import types as t

ASCII_LOWERCASE_WITH_DIGITS: Final = string.ascii_lowercase + string.digits

settings.register_profile(
    name="default",
    parent=settings.get_profile("default"),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,
)


def full_config_strategy() -> st.SearchStrategy[sch.Config]:
    # Preprocessing
    outlier_strategy = st.builds(
        sch._OutlierConfig, remove=st.booleans(), method=st.sampled_from(list(t.OutlierFindMethod))
    )
    preprocessing_strategy = st.builds(
        sch._PreprocessingConfig,
        impute_strategy=st.sampled_from(list(t.ImputeStrategy)),
        scaling_method=st.sampled_from(list(t.ScalingMethod)),
        outlier=outlier_strategy,
    )

    # Features
    features_strategy = st.builds(
        sch._FeatureConfig,
        max_features=st.integers(min_value=1, max_value=20),
        selection_method=st.sampled_from(list(t.FeatureSelectionMethod)),
        correlation_threshold=st.floats(min_value=0.7, max_value=0.9),
    )

    # Model
    model_strategy = st.builds(
        sch._ModelConfig,
        name=st.text(alphabet=ASCII_LOWERCASE_WITH_DIGITS, min_size=5, max_size=20),
        model_type=st.sampled_from(list(t.ModelType)),
        random_state=st.integers(min_value=0, max_value=c.MAX_RANDOM_STATE),
    )

    return st.builds(
        sch.Config,
        preprocessing=preprocessing_strategy,
        features=features_strategy,
        model=model_strategy,
    )


def required_config_strategy() -> st.SearchStrategy[sch.Config]:
    return st.builds(
        sch.Config,
        model=st.builds(
            sch._ModelConfig,
            name=st.text(alphabet=ASCII_LOWERCASE_WITH_DIGITS, min_size=5, max_size=20),
        ),
    )
