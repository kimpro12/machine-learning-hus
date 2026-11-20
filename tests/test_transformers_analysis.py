"""Tests for the custom transformers (transformers.py) and data-analysis helpers (analysis.py)."""
import numpy as np
import pandas as pd

from phishing_ml.analysis import _profile_text, prep_pipeline
from phishing_ml.features import PHISHVN_FEATURES
from phishing_ml.transformers import CorrelationFilter, KMeansOneHot, Winsorizer, resample_indices


def test_winsorizer_learns_on_fit_only():
    train = np.arange(101, dtype=float).reshape(-1, 1)        # 0..100
    w = Winsorizer(0.05, 0.95).fit(train)
    out = w.transform(np.array([[-50.0], [50.0], [500.0]]))
    assert out[0, 0] == w.lo_[0] and out[2, 0] == w.hi_[0] and out[1, 0] == 50.0
    assert w.lo_[0] == 5.0 and w.hi_[0] == 95.0


def test_correlation_filter_drops_redundant_column():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.column_stack([a, 2 * a + 0.001 * rng.normal(size=200), rng.normal(size=200)])
    cf = CorrelationFilter(0.9).fit(X)
    assert cf.keep_.tolist() == [0, 2]
    assert cf.transform(X).shape == (200, 2)


def test_kmeans_onehot_appends_one_cluster_column():
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10])
    out = KMeansOneHot(2, 0).fit(X).transform(X)
    assert out.shape == (20, 4)
    assert np.all(out[:, 2:].sum(1) == 1)
    assert len({tuple(r) for r in out[:, 2:]}) == 2


def test_resample_indices_balance():
    y = np.array([0] * 10 + [1] * 30)
    under = resample_indices(y, 'under')
    over = resample_indices(y, 'over')
    assert np.bincount(y[under]).tolist() == [10, 10]
    assert np.bincount(y[over]).tolist() == [30, 30]
    assert len(set(under)) == 20  # under-sampling never duplicates


def test_prep_pipeline_fits_on_train_only():
    tr = pd.DataFrame({c: [0.0, 1.0, 3.0] for c in PHISHVN_FEATURES})
    te = pd.DataFrame({c: [1000.0] for c in PHISHVN_FEATURES})
    pp = prep_pipeline().fit(tr)
    mean_before = pp.named_steps['scale'].mean_.copy()
    pp.transform(te)
    assert np.allclose(pp.named_steps['scale'].mean_, mean_before)
    assert np.allclose(mean_before, np.log1p([0.0, 1.0, 3.0]).mean())


def test_profile_text_mentions_www_and_query():
    overall = pd.Series({'digit_cnt_q75': 1, 'dash_cnt_q75': 0, 'url_len_q75': 20, 'url_len_q25': 10})
    row = pd.Series({'www_rate': 0.9, 'vn_rate': 0.0, 'median_path_len': 0, 'median_query_len': 5,
                     'median_digit_cnt': 3, 'median_dash_cnt': 0, 'median_url_len': 30})
    t = _profile_text(row, overall)
    assert 'có `www.`' in t and 'tham số truy vấn' in t and 'URL dài' in t and 'nhiều chữ số' in t
