"""Tests for PhishVN-compatible lexical extractor (src/phishing_ml/features.py)."""
import math
from phishing_ml.features import phishvn_features, PHISHVN_FEATURES


def test_feature_count_and_no_https():
    assert len(PHISHVN_FEATURES) == 21
    assert 'is_https' not in PHISHVN_FEATURES


def test_scheme_stripped_basis():
    f = phishvn_features('http://mst.gov.vn')
    assert f['url_len'] == 10.0  # 'mst.gov.vn', not 17 with scheme
    assert f['dom_len'] == 10.0
    assert f['tld_len'] == 6.0  # public suffix 'gov.vn' per dataset
    assert f['subdom_cnt'] == 0.0


def test_private_psl_off_matches_release():
    # Release treats vercel.app-style hosts with suffix 'app' (private domains OFF).
    f = phishvn_features('http://nhantienquoctej13.vercel.app')
    assert f['dom_len'] == 10.0  # 'vercel.app'
    assert f['tld_len'] == 3.0  # 'app', not 'vercel.app'
    assert f['subdom_cnt'] == 1.0


def test_vn_city_suffix_collapses():
    # Newer-PSL city .vn suffixes collapse to 'vn' per dataset snapshot.
    f = phishvn_features('http://x.danang.vn')
    assert f['tld_len'] == 2.0
    assert f['dom_len'] == len('danang.vn')


def test_unknown_suffix_convention():
    # No recognisable suffix: dom = full host, tld empty, subdom = labels - 1.
    f = phishvn_features('http://techcombank.com-ut.xvz')
    assert f['tld_len'] == 0.0
    assert f['dom_len'] == 22.0
    assert f['subdom_cnt'] == 2.0


def test_trailing_slash_excluded_from_path():
    a = phishvn_features('http://example.com/a/')
    b = phishvn_features('http://example.com/a')
    assert a['path_len'] == b['path_len'] == 1.0


def test_empty_and_none_safe():
    for u in ('', None, 'not a url'):
        f = phishvn_features(u)
        assert len(f) == 21
        for v in f.values():
            assert math.isfinite(float(v))


def test_counts_ratios_entropy_consistent():
    f = phishvn_features('http://g88.fun')
    assert f['letter_cnt'] == 4.0
    assert f['digit_cnt'] == 2.0
    assert f['special_cnt'] == 1.0
    assert f['letter_ratio'] == 4.0 / 7.0
    assert f['url_len'] == f['letter_cnt'] + f['digit_cnt'] + f['special_cnt']
