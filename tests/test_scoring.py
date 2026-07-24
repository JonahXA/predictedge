import numpy as np
import pytest

from predictedge.models.baseline import bin_prob, bin_probs
from predictedge.scoring import brier, logloss


def test_brier_certain_forecast_is_zero():
    assert brier(np.array([0.0, 1.0, 0.0]), 1) == 0.0


def test_brier_worst_case():
    assert brier(np.array([1.0, 0.0]), 1) == pytest.approx(2.0)


def test_logloss_of_confident_correct_is_small():
    assert logloss(np.array([0.01, 0.99]), 1) == pytest.approx(-np.log(0.99))


def test_bin_partition_sums_to_one():
    # bottom tail (<80), bins 80-81, 82-83, 84-85, top tail (>85)
    bins = [
        ("less", np.nan, 80.0),
        ("between", 80.0, 81.0),
        ("between", 82.0, 83.0),
        ("between", 84.0, 85.0),
        ("greater", 85.0, np.nan),
    ]
    raw = sum(bin_prob(82.0, 3.0, st, fl, cp) for st, fl, cp in bins)
    assert raw == pytest.approx(1.0, abs=1e-6)
    assert bin_probs(82.0, 3.0, bins).sum() == pytest.approx(1.0)


def test_continuity_correction_directions():
    # yes iff value > 85 means >= 86; with mu far above, prob ~ 1
    assert bin_prob(95, 1.0, "greater", 85, np.nan) > 0.999
    # yes iff value < 80 means <= 79; with mu far above, prob ~ 0
    assert bin_prob(95, 1.0, "less", np.nan, 80) < 0.001
