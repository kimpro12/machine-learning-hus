import pandas as pd
from phishing_ml.split import dedup_and_group, domain_split_ids, random_split_ids


def _df(n=60):
    urls = [f"https://site{i}.com/" for i in range(n // 2)] + [f"https://phish{i}.net/login?x={i}" for i in range(n // 2)]
    labels = [1] * (n // 2) + [0] * (n // 2)
    y = [0] * (n // 2) + [1] * (n // 2)
    return pd.DataFrame({"URL": urls, "label": labels, "y": y, "x": [float(i) for i in range(n)]})


def test_no_overlap_random():
    df, _, _ = dedup_and_group(_df())
    y = df["y"].reset_index(drop=True)
    ids = random_split_ids(y, seed=42)
    assert not set(ids["train"]) & set(ids["val"])
    assert not set(ids["train"]) & set(ids["test"])
    assert len(ids["train"]) + len(ids["val"]) + len(ids["test"]) == len(df)


def test_domain_no_overlap():
    df, _, _ = dedup_and_group(_df())
    y = df["y"].reset_index(drop=True)
    g = df["group"].reset_index(drop=True)
    ids = domain_split_ids(g, y, seed=42)

    garr = g.to_numpy()
    gt = set(garr[ids["train"]])
    gv = set(garr[ids["val"]])
    ge = set(garr[ids["test"]])
    assert not (gt & gv) and not (gt & ge) and not (gv & ge)


def test_conflict_quarantine():
    df = pd.DataFrame({
        "URL": ["https://a.com/", "https://A.COM/", "https://b.com/"],
        "label": [0, 1, 1],
        "y": [1, 0, 0],
        "x": [1.0, 1.0, 2.0],
    })
    kept, quar, stats = dedup_and_group(df)
    assert stats["n_conflict_urls"] >= 1
    assert len(quar) >= 2
    assert "https://a.com/" not in kept["normalized_url"].tolist()
