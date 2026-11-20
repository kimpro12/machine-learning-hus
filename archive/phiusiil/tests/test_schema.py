import pandas as pd
from phishing_ml.schema import validate_schema

def _good_df():
    return pd.DataFrame({"URL": ["https://a.com/", "https://b.com/x"], "label": [0, 1], "URLLength": [10, 20]})

def test_schema_ok():
    info = validate_schema(_good_df())
    assert "errors" not in info
    assert info["n_rows"] == 2

def test_schema_missing_url():
    df = pd.DataFrame({"label": [0, 1]})
    info = validate_schema(df)
    assert "errors" in info

def test_schema_bad_label():
    df = pd.DataFrame({"URL": ["a", "b"], "label": [0, 2]})
    info = validate_schema(df)
    assert "errors" in info

def test_no_fake_fallback():
    # source invalid must raise / report errors, never invent accuracy
    df = pd.DataFrame({"URL": ["x"], "label": [5]})
    info = validate_schema(df)
    assert "errors" in info
