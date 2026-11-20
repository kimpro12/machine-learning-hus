"""Preprocessing helpers: train-only fitting enforced via sklearn Pipelines."""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_xy(df: pd.DataFrame, features: list[str], target: str = "y"):
    X = df[features].copy()
    y = df[target].to_numpy()
    return X, y


def make_pipeline(model, needs_scaling: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)
