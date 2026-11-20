import pandas as pd
from phishing_ml.split import random_split_ids, split_hash

def test_split_determinism():
    y = pd.Series([0, 1] * 50)
    a = random_split_ids(y, seed=42)
    b = random_split_ids(y, seed=42)
    assert a == b
    assert split_hash(a) == split_hash(b)

def test_counts_sum():
    y = pd.Series([0] * 60 + [1] * 40)
    r = random_split_ids(y, seed=42)
    assert sum(len(r[k]) for k in ("train", "val", "test")) == 100
