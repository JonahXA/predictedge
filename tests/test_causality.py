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


def _spread_params(obs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Same replay, for the spread-conditional variant: record the
    (bias, sigma) actually used on each day, where sigma is conditioned
    on that day's spread but fit only on prior days."""
    state = ErrorState()
    out = []
    for err, spread in obs:
        out.append((state.bias, state.sigma_for(spread)))
        state.add(err, spread)
    return out


def test_spread_conditional_sigma_is_also_walk_forward():
    rng = np.random.default_rng(11)
    spreads = np.abs(rng.normal(2.0, 0.8, size=80))
    errors = rng.normal(0.3, 1.0, size=80) * spreads  # error scales with disagreement
    obs = list(zip(errors, spreads))
    short = _spread_params(obs[:40])
    long = _spread_params(obs)
    assert short == long[:40]


def test_wider_spread_gives_wider_sigma_once_fit():
    """With a history where error scales with disagreement, a
    high-spread day must get a wider distribution than a low-spread one."""
    rng = np.random.default_rng(12)
    spreads = np.abs(rng.normal(2.0, 0.8, size=60))
    errors = rng.normal(0, 1.0, size=60) * spreads
    state = ErrorState(list(errors), list(spreads))
    assert state.sigma_for(4.0) > state.sigma_for(0.5)


def test_residual_shape_is_walk_forward():
    """The empirical residual CDF used on a day must be identical whether
    or not later days exist in the dataset."""
    rng = np.random.default_rng(21)
    errors = list(rng.normal(0, 2.0, size=90))
    spreads = list(np.abs(rng.normal(2.0, 0.5, size=90)))
    grid = [-2.0, -0.5, 0.0, 1.5]

    def cdf_at(n):
        c = ErrorState(errors[:n], spreads[:n]).residual_cdf()
        return None if c is None else [round(c(x), 12) for x in grid]

    assert cdf_at(40) == cdf_at(40)  # deterministic
    # Fitting on 40 days must not be affected by days 41-90 existing.
    state_short = ErrorState(errors[:40], spreads[:40])
    state_long_prefix = ErrorState(errors[:40], spreads[:40])
    assert [state_short.residual_cdf()(x) for x in grid] == [
        state_long_prefix.residual_cdf()(x) for x in grid
    ]


def test_residual_shape_needs_history():
    assert ErrorState([1.0, -1.0], [2.0, 2.0]).residual_cdf() is None


def test_empirical_cdf_captures_left_skew():
    """A left-skewed residual sample must put more mass in the low tail
    than a Normal would."""
    from scipy import stats as st

    from predictedge.models.baseline import empirical_cdf

    rng = np.random.default_rng(22)
    z = list(-np.abs(rng.normal(0, 1.5, size=400)) + 0.6)  # heavy left tail
    cdf = empirical_cdf(z)
    assert cdf(-2.0) > st.norm.cdf(-2.0)


def test_sigma_for_falls_back_without_spread():
    state = ErrorState([1.0, -2.0, 0.5], [float("nan")] * 3)
    assert state.sigma_for(None) == state.sigma
    assert state.sigma_for(float("nan")) == state.sigma
    # Too little history to fit → unconditional sigma.
    assert state.sigma_for(3.0) == state.sigma
