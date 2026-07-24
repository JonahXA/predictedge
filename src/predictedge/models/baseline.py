"""Baseline probabilistic model for daily-high markets.

The day's high is modeled as Normal(mu, sigma):

  mu    = day-ahead NWP forecast + a walk-forward bias correction
          (shrunk mean of this city's past forecast errors),
  sigma = walk-forward spread of those errors, shrunk toward a prior
          (PRIOR_SIGMA_F with PRIOR_N pseudo-observations) so early
          events aren't overfit to a handful of residuals.

Bin probabilities integrate this distribution over each strike bin with
a continuity correction (official highs settle on whole degrees F).

Causality note: the error history for event day D uses only days
strictly before D. A day's high is fully determined and publicly
observable (hourly NWS/METAR obs) by local midnight, hours before the
09:00 UTC snapshot, so those errors are legitimately known at decision
time.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..config import PRIOR_N, PRIOR_SIGMA_F


class ErrorState:
    """Walk-forward accumulator of (official_high - forecast) errors for
    one city. Pure function of the errors added so far."""

    def __init__(self, errors: list[float] | None = None) -> None:
        self.errors: list[float] = list(errors) if errors else []

    def add(self, error: float) -> None:
        self.errors.append(error)

    @property
    def bias(self) -> float:
        # Shrunk toward 0 by PRIOR_N pseudo-observations.
        return sum(self.errors) / (len(self.errors) + PRIOR_N)

    @property
    def sigma(self) -> float:
        b = self.bias
        ss = sum((e - b) ** 2 for e in self.errors)
        return math.sqrt((PRIOR_N * PRIOR_SIGMA_F**2 + ss) / (PRIOR_N + len(self.errors)))


def bin_prob(mu: float, sigma: float, strike_type: str, floor: float, cap: float) -> float:
    """P(bin) under Normal(mu, sigma), continuity-corrected for
    integer-degree settlement."""
    z = lambda x: stats.norm.cdf((x - mu) / sigma)
    if strike_type == "greater":  # yes iff value > floor, i.e. >= floor+1
        return 1 - z(floor + 0.5)
    if strike_type == "less":  # yes iff value < cap, i.e. <= cap-1
        return z(cap - 0.5)
    if strike_type == "between":  # yes iff floor <= value <= cap
        return z(cap + 0.5) - z(floor - 0.5)
    raise ValueError(f"unknown strike_type {strike_type}")


def bin_probs(mu: float, sigma: float, bins: list[tuple[str, float, float]]) -> np.ndarray:
    """Probability vector over an event's bins, renormalized (the bins
    partition the line, so the sum is ~1 before rounding)."""
    p = np.array([bin_prob(mu, sigma, st, fl, cp) for st, fl, cp in bins])
    p = np.clip(p, 1e-9, None)
    return p / p.sum()
