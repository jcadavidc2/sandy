"""Pure-function tests for the empirical-CDF calibration of MLB totals
(sandy/over_under/calibration.py). No database needed."""
from __future__ import annotations

import numpy as np
import pytest

from sandy.over_under.calibration import (
    MIN_RESIDUALS,
    P_CEIL,
    P_FLOOR,
    calibrated_probabilities,
)
from sandy.over_under.predictor import compute_over_under_probabilities
from sandy.over_under.schemas import STANDARD_THRESHOLDS


def _resid(n=5000, seed=0, skew=True):
    rng = np.random.default_rng(seed)
    if skew:
        # right-skewed integer-ish errors like real run totals (mean ~0, sd ~4.5)
        r = rng.gamma(shape=4.0, scale=2.25, size=n) - 9.0
    else:
        r = rng.normal(0.0, 4.5, size=n)
    return np.sort(np.round(r))


def test_returns_none_below_minimum_sample():
    assert calibrated_probabilities(9.0, _resid(MIN_RESIDUALS - 1)) is None
    assert calibrated_probabilities(9.0, _resid(MIN_RESIDUALS)) is not None


def test_monotone_non_increasing_across_lines_and_bounded():
    p = calibrated_probabilities(8.9, _resid())
    vals = [p[t] for t in STANDARD_THRESHOLDS]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert all(P_FLOOR <= v <= P_CEIL for v in vals)


def test_matches_empirical_frequency_exactly():
    rs = _resid(2000)
    e = 8.7
    p = calibrated_probabilities(e, rs)
    for t in STANDARD_THRESHOLDS:
        emp = float(np.mean(rs > t - e))
        assert p[t] == pytest.approx(min(max(emp, P_FLOOR), P_CEIL), abs=1e-12)


def test_recovers_normal_when_errors_really_are_normal():
    rs = np.sort(np.random.default_rng(1).normal(0.0, 4.5, size=200_000))
    p = calibrated_probabilities(9.0, rs)
    ref = compute_over_under_probabilities(9.0, residual_std=4.5)
    for t in STANDARD_THRESHOLDS:
        assert p[t] == pytest.approx(ref[t], abs=0.01)


def test_skewed_errors_pull_over_probability_below_the_normal():
    """The whole point: with right-skewed errors (median < mean) the Normal
    over-states P(over) at lines just below the mean."""
    rs = _resid(50_000)
    e = 9.0
    p = calibrated_probabilities(e, rs)
    ref = compute_over_under_probabilities(e, residual_std=float(rs.std()))
    assert p[8.5] < ref[8.5]
    assert p[7.5] < ref[7.5]


def test_higher_expected_total_never_lowers_any_probability():
    rs = _resid()
    lo = calibrated_probabilities(8.0, rs)
    hi = calibrated_probabilities(10.0, rs)
    assert all(hi[t] >= lo[t] for t in STANDARD_THRESHOLDS)
