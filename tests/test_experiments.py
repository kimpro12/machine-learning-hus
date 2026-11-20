"""Tests for the supplementary LogReg experiments (src/phishing_ml/experiments.py)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from phishing_ml.experiments import (
    _macro_f1_counts, add_nowww, best_f1_threshold, build_variants, make_x,
    mcnemar_exact, metrics, paired_bootstrap_macro_f1_diff, strip_www,
)
from phishing_ml.features import PHISHVN_FEATURES, phishvn_features


def test_strip_www_only_leading_host_prefix():
    assert strip_www('https://www.Example.com/a') == 'Example.com/a'
    assert strip_www('http://example.com/www.x') == 'example.com/www.x'
    assert strip_www('wwwexample.com') == 'wwwexample.com'


def test_add_nowww_recomputes_features():
    df = pd.DataFrame({'url': ['https://www.abc.vn/x', 'abc.vn/x']})
    out = add_nowww(df)
    assert out['url_nw'].tolist() == ['abc.vn/x', 'abc.vn/x']
    # both rows map to the same site -> identical www-normalised features
    a = out.loc[0, [f'nw_{c}' for c in PHISHVN_FEATURES]].to_numpy(dtype=float)
    b = out.loc[1, [f'nw_{c}' for c in PHISHVN_FEATURES]].to_numpy(dtype=float)
    assert np.allclose(a, b)
    assert out.loc[0, 'nw_url_len'] == phishvn_features('abc.vn/x')['url_len']


def test_make_x_views():
    df = add_nowww(pd.DataFrame({'url': ['http://www.a.vn/p']}).assign(
        **{c: 0 for c in PHISHVN_FEATURES}))
    v = {d['name']: d for d in build_variants()}
    assert list(make_x(df, v['num_standard']).columns) == PHISHVN_FEATURES
    assert make_x(df, v['url_tfidf']).tolist() == ['www.a.vn/p']
    assert make_x(df, v['url_tfidf_nowww']).tolist() == ['a.vn/p']
    both = make_x(df, v['url_tfidf+num_log1p_nowww'])
    assert both['url_text'].tolist() == ['a.vn/p'] and both['url_len'].iloc[0] == df['nw_url_len'].iloc[0]


def test_variant_names_unique_and_baseline_first():
    names = [v['name'] for v in build_variants()]
    assert names[0] == 'num_standard' and len(names) == len(set(names))


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    m = metrics(y, s, 0.5)
    assert (m['TN'], m['FP'], m['FN'], m['TP']) == (1, 1, 1, 1)
    assert 0 < best_f1_threshold(y, s) <= 0.4  # catches both positives


def test_vectorised_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, (5, 50)), rng.integers(0, 2, (5, 50))
    ref = [f1_score(y[i], p[i], average='macro') for i in range(5)]
    assert np.allclose(_macro_f1_counts(y, p), ref)


def test_mcnemar_and_paired_bootstrap():
    y = np.array([1] * 50 + [0] * 50)
    base = y.copy()
    base[:10] = 0  # baseline misses 10 positives
    n01, n10, p = mcnemar_exact(y, y, base)
    assert (n01, n10) == (10, 0) and p < 0.01
    lo, hi = paired_bootstrap_macro_f1_diff(y, y, base, n_boot=200)
    assert 0 < lo <= hi
    lo, hi = paired_bootstrap_macro_f1_diff(y, base, base, n_boot=50)
    assert lo == hi == 0.0
