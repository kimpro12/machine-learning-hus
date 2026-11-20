"""Schema validation and feature-set definitions."""
from __future__ import annotations

import pandas as pd

TEXT_META = ["FILENAME", "URL", "Domain", "TLD", "Title"]
SUSPECT_DERIVED = [
    "URLSimilarityIndex",
    "TLDLegitimateProb",
    "URLCharProb",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
]
TARGET_ORIG = "label"


def recode_label(s: pd.Series) -> pd.Series:
    """Original: 0=phishing, 1=legitimate. Internal: y=1 phishing, y=0 legitimate."""
    return (s == 0).astype(int)


def validate_schema(df: pd.DataFrame) -> dict:
    errors = []
    if "URL" not in df.columns:
        errors.append("missing URL column")
    if "label" not in df.columns:
        errors.append("missing label column")
    info: dict = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }
    if errors:
        info["errors"] = errors
        return info
    vals = set(pd.unique(df["label"].dropna()).tolist())
    info["label_values"] = sorted(vals)
    if not vals.issubset({0, 1}):
        errors.append(f"label values must be subset of {{0,1}}, got {vals}")
    if len(vals) < 2:
        errors.append("label must contain both classes")
    # NaN / Inf / blank / constant checks
    info["nan_total"] = int(df.isna().sum().sum())
    info["nan_by_col"] = {c: int(v) for c, v in df.isna().sum().items() if int(v) > 0}
    # Inf in numeric
    num = df.select_dtypes(include="number")
    import numpy as np

    info["inf_total"] = int(np.isinf(num.to_numpy(dtype=float, copy=False)).sum()) if num.shape[1] else 0
    # blank strings in object cols
    blanks = {}
    for c in df.select_dtypes(include="object").columns:
        blanks[c] = int((df[c].astype(str).str.strip() == "").sum())
    info["blank_by_col"] = blanks
    # constant features
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    info["constant_cols"] = const
    # duplicated col names
    info["duplicated_colnames"] = bool(len(set(df.columns)) != len(df.columns))
    # class counts
    vc = df["label"].value_counts().to_dict()
    info["label_counts_orig"] = {int(k): int(v) for k, v in vc.items()}
    y = recode_label(df["label"])
    info["y_phishing_share"] = float(y.mean())
    info["y_counts_internal"] = {int(k): int(v) for k, v in y.value_counts().to_dict().items()}
    if errors:
        info["errors"] = errors
    return info


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(TEXT_META + [TARGET_ORIG, "y", "row_id", "normalized_url", "group"])
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def feature_sets(df_columns: list[str], strict_cols: list[str], no_cues_cols: list[str]) -> dict[str, list[str]]:
    full = [c for c in df_columns if c not in set(TEXT_META + [TARGET_ORIG, "y", "row_id", "normalized_url", "group"])]
    # full should be numeric only; filter non-numeric defensively by caller
    no_suspect = [c for c in full if c not in set(SUSPECT_DERIVED)]
    return {
        "FULL_NUMERIC": full,
        "NO_SUSPECT_DERIVED": no_suspect,
        "URL_LEXICAL_STRICT": list(strict_cols),
        "STRICT_NO_COLLECTION_CUES": list(no_cues_cols),
    }
