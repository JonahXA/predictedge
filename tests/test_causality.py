"""Feature causality: the model parameters used for an event must not
change when later events are added to the dataset.

The walk-forward state (bias, sigma) for day D is a pure function of
errors from days strictly before D — so replaying a longer history must
reproduce byte-identical parameters for every earlier day.
"""

import numpy as np

from predictedge.models.baseline import ErrorState


def _params_by_day(errors_by_day: list[float]) -> list[tuple[float, float]]:
    """Simulate the backtest loop: record (bias, sigma) as used for each
    day, updating the state only after the day is scored."""
    state = ErrorState()
    out = []
    for e in errors_by_day:
        out.append((state.bias, state.sigma))  # parameters used at decision time
        state.add(e)  # observed only after the event
    return out


def test_adding_future_days_never_changes_past_features():
    rng = np.random.default_rng(7)
    errors = list(rng.normal(0.5, 2.5, size=60))
    short = _params_by_day(errors[:30])
    long = _params_by_day(errors)  # 30 more future days appended
    assert short == long[:30]


def test_first_event_uses_pure_prior():
    from predictedge.config import PRIOR_SIGMA_F

    state = ErrorState()
    assert state.bias == 0.0
    assert state.sigma == PRIOR_SIGMA_F
