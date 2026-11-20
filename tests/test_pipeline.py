"""Smoke tests for the PhishVN pipeline (src/phishing_ml/pipeline.py)."""
import numpy as np
from phishing_ml.pipeline import grids, norm_key, _slice_frame, _mrow


def test_norm_key():
    assert norm_key('http://Example.COM/a/') == 'example.com/a'
    assert norm_key('https://EXAMPLE.com') == 'example.com'
    assert norm_key('example.com/') == 'example.com'
    assert norm_key('  HTTP://A.vn/X ') == 'a.vn/x'


def test_grids_spec_stable():
    g = grids()
    names = [m[0] for m in g]
    assert names == ['dummy_most_frequent',
                     'logreg_C0.1', 'logreg_C1', 'logreg_C10',
                     'rf_mdNone_msl1', 'rf_md16_msl1', 'rf_mdNone_msl3',
                     'linsvc_C0.1', 'linsvc_C1', 'linsvc_C10']
    assert len(names) == 10


def test_slice_frame_flags_tiny_n():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    s = np.array([0.1, 0.9, 0.8, 0.7])
    r = _slice_frame(y, p, s, 'x', 'FULL')
    assert r['n'] == 4 and r['tiny_n_flag'] is True
    assert r['accuracy'] == 0.75


def test_mrow_keys():
    from phishing_ml.evaluate import compute_metrics
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 0, 1])
    m = compute_metrics(y, p, np.array([0.1, 0.9, 0.2, 0.8]))
    r = _mrow(m, y, p, 0.2, 0.9)
    for k in ('accuracy', 'bal_acc', 'prec', 'rec', 'spec', 'f1', 'macro_f1',
              'roc', 'pr', 'TN', 'FP', 'FN', 'TP', 'acc_lo', 'acc_hi'):
        assert k in r
