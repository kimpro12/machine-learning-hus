import pandas as pd
from phishing_ml.schema import recode_label

def test_label_mapping():
    s = pd.Series([0, 1, 0, 1])
    y = recode_label(s)
    assert y.tolist() == [1, 0, 1, 0]
    # internal: y=1 phishing
    assert y.sum() == 2
