import math
from phishing_ml.url_features import extract_url_features, shannon_entropy, STRICT_COLS

def test_parser_https_www():
    f = extract_url_features("https://www.example.org/a?x=1")
    assert f["is_https"] == 1.0
    assert f["has_www"] == 1.0
    assert f["path_present"] == 1.0
    assert f["query_present"] == 1.0
    assert f["num_query_params"] == 1.0
    for c in STRICT_COLS:
        assert c in f
        assert math.isfinite(f[c])

def test_ip_punycode_port():
    f = extract_url_features("http://192.168.0.1:8080/xn--test")
    assert f["host_is_ip"] == 1.0
    assert f["has_port"] == 1.0
    f2 = extract_url_features("https://xn--exmple-cua.com/")
    assert f2["has_punycode"] == 1.0

def test_malformed():
    f = extract_url_features("not a url :::")
    assert math.isfinite(f["url_length"])
    assert f["parse_error_flag"] in (0.0, 1.0)
    for v in f.values():
        assert math.isfinite(float(v))

def test_entropy():
    assert shannon_entropy("") == 0.0
    assert abs(shannon_entropy("aa") - 0.0) < 1e-9
    assert shannon_entropy("ab") > 0.9

def test_deterministic():
    a = extract_url_features("https://www.example.org/a?x=1")
    b = extract_url_features("https://www.example.org/a?x=1")
    assert a == b
