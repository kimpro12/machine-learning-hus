"""Tests for exact Clopper-Pearson CI (src/phishing_ml/evaluate.py)."""
import math
from phishing_ml.evaluate import clopper_pearson_ci, bootstrap_ci


def test_perfect_accuracy_rule_of_three():
    lo, hi = clopper_pearson_ci(47030, 47030)
    assert hi == 1.0
    assert abs(lo - 0.025 ** (1.0 / 47030)) < 1e-12
    assert abs(lo - 0.9999215663) < 1e-6


def test_zero_successes():
    lo, hi = clopper_pearson_ci(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05


def test_middle_matches_scipy():
    from scipy.stats import beta as beta_dist
    lo, hi = clopper_pearson_ci(8070, 8941)
    assert abs(lo - float(beta_dist.ppf(0.025, 8070, 8941 - 8070 + 1))) < 1e-12
    assert abs(hi - float(beta_dist.ppf(0.975, 8071, 8941 - 8070))) < 1e-12


def test_empty():
    lo, hi = clopper_pearson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_flags_degenerate_perfect():
    out = bootstrap_ci([1] * 20 + [0] * 20, [1] * 20 + [0] * 20, n_boot=50, seed=42)
    assert out['acc_ci95'] == (1.0, 1.0)  # percentile degenerate...
    assert out['acc_exact_ci95'][1] == 1.0
    assert out['acc_exact_ci95'][0] < 1.0  # ...exact CI is not
    assert 'degenerate' in out.get('acc_ci95_note', '')
