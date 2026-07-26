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

from ..config import (
    MAX_SIGMA_F,
    MAX_SPREAD_SLOPE,
    MIN_N_SHAPE_FIT,
    MIN_N_SPREAD_FIT,
    MIN_SIGMA_F,
    PRIOR_N,
    PRIOR_SIGMA_F,
)


class ErrorState:
    """Walk-forward accumulator of (official_high - forecast) errors for
    one city, optionally paired with that day's inter-model spread.
    Pure function of the observations added so far."""

    def __init__(self, errors: list[float] | None = None,
                 spreads: list[float] | None = None) -> None:
        self.errors: list[float] = list(errors) if errors else []
        self.spreads: list[float] = list(spreads) if spreads else []

    def add(self, error: float, spread: float | None = None) -> None:
        self.errors.append(error)
        self.spreads.append(float("nan") if spread is None else spread)

    @property
    def bias(self) -> float:
        # Shrunk toward 0 by PRIOR_N pseudo-observations.
        return sum(self.errors) / (len(self.errors) + PRIOR_N)

    @property
    def sigma(self) -> float:
        b = self.bias
        ss = sum((e - b) ** 2 for e in self.errors)
        return math.sqrt((PRIOR_N * PRIOR_SIGMA_F**2 + ss) / (PRIOR_N + len(self.errors)))

    def sigma_for(self, spread: float | None) -> float:
        """Sigma conditioned on today's inter-model disagreement.

        Regresses past squared errors on past squared spreads
        (var = a + b*spread^2, both coefficients constrained non-negative)
        and shrinks the fitted variance toward the unconditional variance
        by PRIOR_N pseudo-observations. Falls back to the unconditional
        sigma when there is no spread or too little history to fit."""
        base = self.sigma
        if spread is None or not math.isfinite(spread):
            return base
        pairs = [(e, s) for e, s in zip(self.errors, self.spreads) if math.isfinite(s)]
        if len(pairs) < MIN_N_SPREAD_FIT:
            return base

        b0 = self.bias
        x = np.array([s**2 for _, s in pairs])
        y = np.array([(e - b0) ** 2 for e, _ in pairs])
        xm, ym = x.mean(), y.mean()
        denom = float(((x - xm) ** 2).sum())
        slope = float(((x - xm) * (y - ym)).sum() / denom) if denom > 0 else 0.0
        slope = min(max(slope, 0.0), MAX_SPREAD_SLOPE)
        intercept = max(ym - slope * xm, MIN_SIGMA_F**2)

        fitted = intercept + slope * spread**2
        n = len(pairs)
        var = (n * fitted + PRIOR_N * base**2) / (n + PRIOR_N)
        return math.sqrt(min(max(var, MIN_SIGMA_F**2), MAX_SIGMA_F**2))

    def residual_cdf(self, spread_sigma: bool = False):
        """Standardized-residual CDF from history, or None when there is
        too little history to estimate a shape (caller falls back to the
        Normal). Each past error is standardized by the sigma that would
        have been used on that day, so the shape is scale-free."""
        if len(self.errors) < MIN_N_SHAPE_FIT:
            return None
        b = self.bias
        z = []
        for i, e in enumerate(self.errors):
            past = ErrorState(self.errors[:i], self.spreads[:i])
            s = past.sigma_for(self.spreads[i]) if spread_sigma else past.sigma
            if s > 0:
                z.append((e - b) / s)
        return empirical_cdf(z) if len(z) >= MIN_N_SHAPE_FIT else None


def empirical_cdf(standardized: list[float], smoothing: float = 0.6):
    """Smoothed CDF of past standardized residuals.

    Forecast errors here are left-skewed and fat-tailed (a day can bust
    cold far more dramatically than it busts warm), which a Normal cannot
    represent. This is a Gaussian kernel density estimate over the
    observed standardized residuals, so skew and kurtosis are carried
    directly from history instead of assumed away."""
    z = np.asarray([v for v in standardized if math.isfinite(v)], dtype=float)

    def cdf(x: float) -> float:
        return float(stats.norm.cdf((x - z) / smoothing).mean())

    return cdf


def bin_prob(mu: float, sigma: float, strike_type: str, floor: float, cap: float,
             cdf=None) -> float:
    """P(bin) under the predictive distribution, continuity-corrected for
    integer-degree settlement. Defaults to Normal(mu, sigma); pass `cdf`
    for a standardized non-Gaussian residual distribution."""
    if cdf is None:
        z = lambda x: stats.norm.cdf((x - mu) / sigma)
    else:
        z = lambda x: cdf((x - mu) / sigma)
    if strike_type == "greater":  # yes iff value > floor, i.e. >= floor+1
        return 1 - z(floor + 0.5)
    if strike_type == "less":  # yes iff value < cap, i.e. <= cap-1
        return z(cap - 0.5)
    if strike_type == "between":  # yes iff floor <= value <= cap
        return z(cap + 0.5) - z(floor - 0.5)
    raise ValueError(f"unknown strike_type {strike_type}")


def bin_probs(mu: float, sigma: float, bins: list[tuple[str, float, float]],
              cdf=None) -> np.ndarray:
    """Probability vector over an event's bins, renormalized (the bins
    partition the line, so the sum is ~1 before rounding)."""
    p = np.array([bin_prob(mu, sigma, st, fl, cp, cdf) for st, fl, cp in bins])
    p = np.clip(p, 1e-9, None)
    return p / p.sum()
