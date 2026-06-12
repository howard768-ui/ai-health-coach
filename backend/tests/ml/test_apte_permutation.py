"""Phase 9A APTE permutation test unit tests.

Tests cover:
1. Known-effect detection (significant p-value)
2. Independent data rejection (non-significant)
3. Effect size (Cohen's d) computation
4. Autocorrelation assessment
5. Edge cases (insufficient data)

Run: ``cd backend && uv run python -m pytest tests/ml/test_apte_permutation.py -v``
"""

from __future__ import annotations

import os


os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

import numpy as np
from ml.discovery.apte import (
    compute_apte_permutation,
    assess_autocorrelation,
)


# ---------------------------------------------------------------------------
# Known-effect detection
# ---------------------------------------------------------------------------


def test_permutation_detects_known_large_effect():
    """When treatment adds a large shift, p-value should be significant."""
    rng = np.random.default_rng(42)
    n = 14
    baseline = rng.normal(50, 5, size=n)
    treatment = rng.normal(60, 5, size=n)  # +10 shift (d ~ 2.0)

    result = compute_apte_permutation(baseline, treatment, n_resamples=999)
    assert result is not None
    assert result.p_value < 0.05, f"Expected significant, got p={result.p_value}"
    assert result.apte > 5, f"Expected APTE > 5, got {result.apte}"
    assert result.effect_size_d > 1.0, f"Expected large d, got {result.effect_size_d}"


def test_permutation_rejects_independent_data():
    """When baseline and treatment are from the same distribution, p > 0.05."""
    rng = np.random.default_rng(42)
    n = 14
    baseline = rng.normal(50, 5, size=n)
    treatment = rng.normal(50, 5, size=n)

    result = compute_apte_permutation(baseline, treatment, n_resamples=999)
    assert result is not None
    assert result.p_value > 0.05, f"Expected non-significant, got p={result.p_value}"


def test_permutation_direction_correct():
    """APTE sign should match the direction of the treatment effect."""
    rng = np.random.default_rng(42)
    n = 14
    baseline = rng.normal(50, 5, size=n)
    treatment_up = rng.normal(60, 5, size=n)
    treatment_down = rng.normal(40, 5, size=n)

    result_up = compute_apte_permutation(baseline, treatment_up, n_resamples=999)
    result_down = compute_apte_permutation(baseline, treatment_down, n_resamples=999)

    assert result_up is not None and result_up.apte > 0
    assert result_down is not None and result_down.apte < 0


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------


def test_cohens_d_correct_for_known_data():
    """Cohen's d should be approximately 1.0 for a 1-SD shift."""
    baseline = np.array([50.0] * 14)
    treatment = np.array([55.0] * 14)
    # Both have SD=0, so pooled_std is 0, d would be inf.
    # Use data with some variance.
    rng = np.random.default_rng(42)
    baseline = rng.normal(50, 5, size=14)
    treatment = baseline + 5  # exact 5-unit shift, same variance

    result = compute_apte_permutation(baseline, treatment, n_resamples=999)
    assert result is not None
    # d should be close to 5/5 = 1.0.
    assert 0.5 < result.effect_size_d < 2.0, f"Expected d near 1.0, got {result.effect_size_d}"


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def test_autocorrelation_near_zero_for_white_noise():
    """White noise should have near-zero lag-1 autocorrelation."""
    rng = np.random.default_rng(42)
    series = rng.normal(0, 1, size=50)
    rho, n_eff = assess_autocorrelation(series)
    assert abs(rho) < 0.3, f"Expected low autocorrelation, got {rho}"
    assert n_eff > 20, f"Expected high effective n, got {n_eff}"


def test_autocorrelation_positive_for_ar1():
    """AR(1) process should have positive lag-1 autocorrelation."""
    rng = np.random.default_rng(42)
    n = 100
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = 0.7 * series[t - 1] + rng.normal(0, 1)

    rho, n_eff = assess_autocorrelation(series)
    assert rho > 0.4, f"Expected positive autocorrelation, got {rho}"
    assert n_eff < n * 0.5, f"Expected reduced effective n, got {n_eff} (full n={n})"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_insufficient_data_returns_none():
    """Fewer than MIN_OBSERVATIONS should return None."""
    baseline = np.array([1.0, 2.0, 3.0])
    treatment = np.array([4.0, 5.0, 6.0])
    result = compute_apte_permutation(baseline, treatment)
    assert result is None


def test_handles_nan_values():
    """NaNs should be dropped before computing APTE."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(50, 5, size=20)
    treatment = rng.normal(60, 5, size=20)
    baseline[::5] = np.nan
    treatment[::7] = np.nan

    result = compute_apte_permutation(baseline, treatment, n_resamples=999)
    assert result is not None
    assert result.baseline_n < 20  # NaNs dropped
    assert result.treatment_n < 20


def test_means_reported_correctly():
    """Baseline and treatment means should be correctly computed."""
    baseline = np.array([10.0] * 10)
    treatment = np.array([20.0] * 10)

    result = compute_apte_permutation(baseline, treatment, n_resamples=99)
    assert result is not None
    assert result.baseline_mean == 10.0
    assert result.treatment_mean == 20.0
    assert result.apte == 10.0
