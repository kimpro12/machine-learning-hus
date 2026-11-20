"""Reporting: SUMMARY.md + missing figures (confusion, ROC/PR, missing, duplicates, correlation)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_missing_fig(df: pd.DataFrame, fig_dir: Path):
    miss = df.isna().sum()
    plt.figure(figsize=(10, 3))
    if (miss > 0).any():
        m = miss[miss > 0].sort_values(ascending=False).head(20)
        plt.bar(m.index.astype(str), m.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("missing count")
        plt.title("Missing values by column (top 20)")
    else:
        plt.text(0.5, 0.5, f"No missing values (n={len(df)})", ha="center", va="center")
        plt.title("Missing values: none detected")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "missing_by_column.png", dpi=150)
    plt.close()


def _ensure_duplicate_fig(stats: dict, fig_dir: Path):
    keys = ["n_exact_dup_rows", "n_dup_norm_urls", "n_conflict_urls", "n_quarantined_rows"]
    vals = [stats.get(k, 0) for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.title("Duplicate / quarantine audit")
    plt.ylabel("count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "duplicate_audit.png", dpi=150)
    plt.close()


def _ensure_corr_fig(audit_dir: Path, fig_dir: Path):
    p = audit_dir / "feature_correlation_top.csv"
    if not p.exists():
        return
    try:
        df = pd.read_csv(p)
        if len(df) == 0:
            return
        plt.figure(figsize=(8, 4))
        plt.barh(df["feature"][::-1].astype(str), df["abs_r"][::-1])
        plt.xlabel("|pearson r| with y (train only)")
        plt.title("Top feature-label correlations (train only)")
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_correlation_top.png", dpi=150)
        plt.close()
    except Exception:
        pass


def _confusion_and_roc(metrics_dir: Path, fig_dir: Path):
    # use final_random_test.csv best strict row's confusion if available else quick
    for src in ["final_random_test.csv", "quick_validation.csv"]:
        p = metrics_dir / src
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            ok = df[df.get("status", "OK") == "OK"] if "status" in df.columns else df
            if len(ok) == 0:
                continue
            # best strict else best overall
            sub = ok[ok["feature_set"] == "URL_LEXICAL_STRICT"] if "feature_set" in ok.columns else ok
            if len(sub) == 0:
                sub = ok
            b = sub.sort_values("f1_phishing", ascending=False).iloc[0]
            tn, fp, fn, tp = int(b.get("TN", 0)), int(b.get("FP", 0)), int(b.get("FN", 0)), int(b.get("TP", 0))
            plt.figure(figsize=(4, 4))
            plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
            plt.xticks([0, 1], ["pred 0", "pred 1"])
            plt.yticks([0, 1], ["true 0", "true 1"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str([[tn, fp], [fn, tp]][i][j]), ha="center", va="center")
            plt.title(f"Confusion best strict\n{b.get('feature_set')}/{b.get('model')}\n[[TN,FP],[FN,TP]] y=1 phishing")
            plt.tight_layout()
            plt.savefig(fig_dir / "confusion_matrix_best_strict.png", dpi=150)
            plt.close()
            break
        except Exception:
            pass
    # ROC/PR placeholder from scores? we lack stored scores; plot metric bars instead if needed
    # ensure roc_pr_best_models.png exists as summary of ROC-AUC/PR-AUC
    try:
        p = metrics_dir / ("final_random_test.csv" if (metrics_dir / "final_random_test.csv").exists() else "quick_validation.csv")
        df = pd.read_csv(p)
        ok = df[df.get("status", "OK") == "OK"] if "status" in df.columns else df
        if len(ok) and "roc_auc" in ok.columns:
            top = ok.sort_values("roc_auc", ascending=False).head(8)
            plt.figure(figsize=(9, 4))
            x = (top["feature_set"].astype(str) + "/" + top["model"].astype(str))
            plt.bar(x, top["roc_auc"].fillna(0).values, alpha=0.7, label="ROC-AUC")
            if "pr_auc" in top.columns:
                plt.bar(x, top["pr_auc"].fillna(0).values, alpha=0.4, label="PR-AUC")
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("AUC")
            plt.title("ROC/PR-AUC top models (validation or final test)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "roc_pr_best_models.png", dpi=150)
            plt.close()
    except Exception:
        pass


def build_report(outputs: Path, decision: dict | None = None):
    fig_dir = outputs / "figures"
    metrics_dir = outputs / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # SUMMARY.md
    qpath = metrics_dir / "quick_validation.csv"
    lines = ["# PhiUSIIL Phishing Detection — Summary", ""]
    lines.append("Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.")
    lines.append("")
    if qpath.exists():
        try:
            q = pd.read_csv(qpath)
            ok = q[q["status"] == "OK"] if "status" in q.columns else q
            lines.append(f"- Quick validation runs: {len(ok)}/{len(q)} OK.")
            for fs in sorted(ok["feature_set"].unique().tolist()) if "feature_set" in ok.columns else []:
                sub = ok[ok["feature_set"] == fs].sort_values("f1_phishing", ascending=False).iloc[0]
                lines.append(f"  - {fs}: best {sub['model']} acc={float(sub['accuracy']):.4f} f1={float(sub['f1_phishing']):.4f} roc_auc={float(sub.get('roc_auc', float('nan'))):.4f}")
            # preliminary conclusion
            lines.append("")
            lines.append("Preliminary: see outputs/metrics/quick_validation.csv and figures quick_accuracy_by_feature_set.png / quick_f1_by_feature_set.png.")
        except Exception as e:
            lines.append(f"- quick parse error: {e}")
    for extra in ["final_random_test.csv", "final_group_test.csv", "domain_gap.csv", "pca_results.csv", "clustering_metrics.csv", "split_sensitivity.csv"]:
        p = metrics_dir / extra
        if p.exists():
            try:
                df = pd.read_csv(p)
                lines.append(f"- {extra}: {len(df)} rows.")
            except Exception:
                pass
    if decision:
        lines.append("")
        lines.append(f"Decision: `{decision.get('status')}` — {decision.get('decision_reason')}")
        if decision.get("collection_bias_flags", {}).get("SEVERE_COLLECTION_BIAS"):
            lines.append("> BENCHMARK_NOT_REAL_WORLD_VALIDATED")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- outputs/audit/, outputs/metrics/, outputs/figures/, outputs/manifests/, outputs/decision.json, outputs/DECISION.md")
    (outputs / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return str(outputs / "SUMMARY.md")
