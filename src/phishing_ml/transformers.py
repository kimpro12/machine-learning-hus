"""Small sklearn-compatible transformers used by the preprocessing experiments.

Every transformer learns its parameters in `fit` (train split only, enforced by
sklearn Pipelines) and applies them unchanged in `transform` (val/test/external).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class Winsorizer(BaseEstimator, TransformerMixin):
    """Outlier handling: clip each column to its [lower, upper] train quantiles.

    Example: if 99% of train URLs have url_len <= 80, a 400-char URL is treated as 80.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Feature selection: drop a column if |Pearson r| with an earlier kept column > threshold.

    Example: letter_cnt vs url_len have r=0.95, so one of them is redundant.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.abs(np.corrcoef(X, rowvar=False))
        r = np.nan_to_num(r)
        keep: list[int] = []
        for j in range(X.shape[1]):
            if all(r[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class KMeansOneHot(BaseEstimator, TransformerMixin):
    """Clustering as a feature: append a one-hot 'which K-Means cluster' block to X.

    The clusters are learned without labels on train; each URL then gets k extra 0/1
    columns saying which "URL type" it belongs to.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.km_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        onehot = np.eye(self.n_clusters)[self.km_.predict(X)]
        return np.hstack([X, onehot])


def resample_indices(y, how: str, seed: int = 42) -> np.ndarray:
    """Row indices for random under-/over-sampling to a 50/50 class balance (train only).

    under: drop random majority rows until both classes have the minority's size.
    over:  duplicate random minority rows until both classes have the majority's size.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    small, big = (idx0, idx1) if len(idx0) < len(idx1) else (idx1, idx0)
    if how == 'under':
        big = rng.choice(big, size=len(small), replace=False)
    elif how == 'over':
        small = np.concatenate([small, rng.choice(small, size=len(big) - len(small), replace=True)])
    else:
        raise ValueError(f'unknown resampling: {how}')
    return np.sort(np.concatenate([small, big]))
