"""Small sklearn-compatible transformers used by the preprocessing experiments.

Every transformer learns its parameters in `fit` (train split only, enforced by
sklearn Pipelines) and applies them unchanged in `transform` (val/test/external).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


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
