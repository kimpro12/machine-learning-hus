"""GO/NO-GO decision (two axes: academic viability vs real-world validity)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def decide(metrics_dir: Path, manifests_dir: Path, cfg: dict, audit_summary: dict, out_path: Path) -> dict:
    qpath = metrics_dir / "quick_validation.csv"
    f_rand = metrics_dir / "final_random_test.csv"
    f_grp = metrics_dir / "final_group_test.csv"
    gap_path = metrics_dir / "domain_gap.csv"
    d = cfg.get("decision", {})
    acc_t = float(d.get("academic_accuracy_target", 0.95))
    f1_t = float(d.get("academic_f1_target", 0.95))
    g_acc_t = float(d.get("robust_group_accuracy_target", 0.90))
    g_f1_t = float(d.get("robust_group_f1_target", 0.90))
    max_gap = float(d.get("max_random_group_gap_pp", 5.0))
    proxy_w = float(d.get("label_proxy_warning", 0.99))

    def best_of(csv: Path):
        if not csv.exists():
            return None
        df = pd.read_csv(csv)
        ok = df[df.get("status", "OK") == "OK"] if "status" in df.columns else df
        if len(ok) == 0:
            return None
        # filter to non-dummy? keep all but prefer by f1
        b = ok.sort_values("f1_phishing", ascending=False).iloc[0].to_dict()
        return b

    # Prefer final test if available else quick validation
    best_full = best_of(f_rand) if f_rand.exists() else None
    # collect per-feature-set bests from final random or quick
    src = f_rand if f_rand.exists() else qpath
    strict_row = clean_row = full_row = None
    if src.exists():
        df = pd.read_csv(src)
        ok = df[df.get("status", "OK") == "OK"] if "status" in df.columns else df
        for fs in ["FULL_NUMERIC", "NO_SUSPECT_DERIVED", "URL_LEXICAL_STRICT", "STRICT_NO_COLLECTION_CUES"]:
            sub = ok[ok.get("feature_set", "") == fs] if "feature_set" in ok.columns else pd.DataFrame()
            if len(sub):
                b = sub.sort_values("f1_phishing", ascending=False).iloc[0].to_dict()
                if fs == "FULL_NUMERIC":
                    full_row = b
                if fs == "NO_SUSPECT_DERIVED":
                    clean_row = b
                if fs == "URL_LEXICAL_STRICT":
                    strict_row = b
    grp_strict = None
    if f_grp.exists():
        df = pd.read_csv(f_grp)
        ok = df[df.get("status", "OK") == "OK"] if "status" in df.columns else df
        sub = ok[ok.get("feature_set", "") == "URL_LEXICAL_STRICT"] if "feature_set" in ok.columns else pd.DataFrame()
        if len(sub):
            grp_strict = sub.sort_values("f1_phishing", ascending=False).iloc[0].to_dict()
    else:
        # fallback domain_gap
        if gap_path.exists():
            try:
                g = pd.read_csv(gap_path)
                r = g[g["feature_set"] == "URL_LEXICAL_STRICT"]
                if len(r):
                    grp_strict = {"accuracy": float(r.iloc[0]["domain_val_acc"]), "f1_phishing": float(r.iloc[0]["domain_val_f1"]), "model": str(r.iloc[0]["model"]), "feature_set": "URL_LEXICAL_STRICT", "source": "domain_val"}
            except Exception:
                pass

    # audit flags
    severe_bias = bool(audit_summary.get("SEVERE_COLLECTION_BIAS", False))
    proxy = bool(audit_summary.get("SUSPECT_LABEL_PROXY", False))
    top_proxy_acc = float(audit_summary.get("top_stump_acc", 0.0) or 0.0)
    rates = audit_summary.get("rates_by_label", {})

    # academic viability: random holdout (final preferred, else validation) on FULL or STRICT reaching targets
    acad = False
    acad_detail = ""
    ref_row = full_row or strict_row or best_full
    if ref_row is not None:
        acad = (float(ref_row.get("accuracy", 0)) >= acc_t and float(ref_row.get("f1_phishing", 0)) >= f1_t)
        acad_detail = f"best random: {ref_row.get('feature_set')}/{ref_row.get('model')} acc={float(ref_row.get('accuracy',0)):.4f} f1={float(ref_row.get('f1_phishing',0)):.4f} vs target {acc_t}/{f1_t}"
    # real-world validity: strict group >= targets and gap small and no severe flags
    real = False
    gap_acc = None
    if strict_row is not None and grp_strict is not None:
        try:
            gap_acc = (float(strict_row.get("accuracy", 0)) - float(grp_strict.get("accuracy", 0))) * 100
        except Exception:
            gap_acc = None
    if grp_strict is not None:
        real = (float(grp_strict.get("accuracy", 0)) >= g_acc_t and float(grp_strict.get("f1_phishing", 0)) >= g_f1_t and not severe_bias and not proxy)
    # gap warning
    gap_warn = gap_acc is not None and abs(gap_acc) > max_gap

    # status priority: BLOCKED > CHANGE > BORDERLINE > KEEP_WITH_WARNINGS > KEEP_ACADEMIC
    # BLOCKED if no data
    if ref_row is None:
        status = "BLOCKED"
        reason = "No model results available; data or training blocked."
    elif float((strict_row or {}).get("accuracy", 0)) < 0.90 and float((strict_row or {}).get("f1_phishing", 0)) < 0.90 and (clean_row is None or (float(clean_row.get("accuracy", 0)) < 0.90 and float(clean_row.get("f1_phishing", 0)) < 0.90)):
        # check bug? assume checked; recommend change
        status = "CHANGE_DATASET_RECOMMENDED"
        reason = f"Best clean/strict random acc<0.90 and F1<0.90 after checks; {acad_detail}."
    elif strict_row is not None and 0.90 <= float(strict_row.get("accuracy", 0)) < 0.95:
        status = "BORDERLINE_REVIEW"
        reason = f"Strict accuracy in 90-95% borderline; {acad_detail}."
    elif acad and (not real or severe_bias or proxy or gap_warn):
        status = "KEEP_WITH_WARNINGS"
        ws = []
        if severe_bias:
            ws.append("SEVERE_COLLECTION_BIAS")
        if proxy:
            ws.append(f"SUSPECT_LABEL_PROXY(top stump acc={top_proxy_acc:.4f}>={proxy_w})")
        if gap_warn:
            ws.append(f"random-domain gap {gap_acc:.2f}pp > {max_gap}pp")
        if grp_strict is not None and (float(grp_strict.get("accuracy", 0)) < g_acc_t or float(grp_strict.get("f1_phishing", 0)) < g_f1_t):
            ws.append(f"domain strict below {g_acc_t}/{g_f1_t}")
        reason = f"Academic targets met ({acad_detail}) but warnings: {'; '.join(ws) if ws else 'real-world validity not established'}. Reframe as leakage/bias case study; do not claim deployment."
    elif acad and real:
        status = "KEEP_ACADEMIC"
        reason = f"Academic and robust targets met. {acad_detail}; domain strict acc={float(grp_strict.get('accuracy')):.4f} f1={float(grp_strict.get('f1_phishing')):.4f}."
    else:
        # acad false but not change-level? borderline
        if ref_row is not None and (float(ref_row.get("accuracy", 0)) >= 0.90):
            status = "BORDERLINE_REVIEW"
            reason = f"Below 95% academic bar; {acad_detail}. Review errors before keep/change."
        else:
            status = "CHANGE_DATASET_RECOMMENDED"
            reason = f"Academic bar not met; {acad_detail}."
    # dataset hash
    ds_hash = ""
    try:
        for mp in [manifests_dir / "data_manifest.json", metrics_dir.parent / "manifests" / "data_manifest.json"]:
            if mp.exists():
                ds_hash = json.loads(mp.read_text()).get("sha256", "")
                break
    except Exception:
        pass
    # random/domain test metrics summary
    def load_csv(p: Path):
        if p.exists():
            try:
                return pd.read_csv(p).to_dict(orient="records")
            except Exception:
                return []
        return []
    decision = {
        "status": status,
        "academic_viability": bool(acad),
        "real_world_validity": bool(real),
        "decision_reason": reason,
        "dataset_hash": ds_hash,
        "best_model_full": {k: full_row.get(k) for k in ["feature_set", "model", "accuracy", "f1_phishing", "roc_auc", "n_train", "n_val", "n_test"] if full_row and k in full_row} if full_row else None,
        "best_model_clean": {k: clean_row.get(k) for k in ["feature_set", "model", "accuracy", "f1_phishing", "roc_auc"] if clean_row and k in clean_row} if clean_row else None,
        "best_model_strict": {k: strict_row.get(k) for k in ["feature_set", "model", "accuracy", "f1_phishing", "roc_auc"] if strict_row and k in strict_row} if strict_row else None,
        "random_test_metrics": load_csv(f_rand)[:5] if f_rand.exists() else load_csv(qpath)[:3],
        "domain_test_metrics": load_csv(f_grp)[:5] if f_grp.exists() else load_csv(gap_path),
        "single_feature_proxy": {"flag": proxy, "top_acc": top_proxy_acc, "top_feature": audit_summary.get("top_stump_feature", "")},
        "collection_bias_flags": {"SEVERE_COLLECTION_BIAS": severe_bias, "rates": rates},
        "recommendation": ("KEEP dataset as academic/leakage-bias case study with reframed RQs; BENCHMARK_NOT_REAL_WORLD_VALIDATED." if status in ("KEEP_ACADEMIC", "KEEP_WITH_WARNINGS") else ("Review borderline then decide." if status == "BORDERLINE_REVIEW" else "Change dataset or re-scope.")),
        "limitations": [
            "FULL_NUMERIC is optimistic (derived/crawled features, corpus-dependent).",
            "URL_LEXICAL_STRICT still benefits from collection bias (homepage vs path).",
            "No independent external labeled set; real-world validity NOT established.",
            "Test holdout opened once post-freeze; no test-set tuning.",
        ],
        "next_steps": [
            "If KEEP: reframe report around leakage/ablation/domain-shift (RQ2-RQ4).",
            "If deployment goal: collect independent URLs withtime-split + external labels.",
            "Add STRICT_NO_COLLECTION_CUES as primary robustness check.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    # DECISION.md with 5 lines on top
    md = _decision_md(decision, strict_row, grp_strict, full_row, audit_summary, metrics_dir)
    (out_path.parent / "DECISION.md").write_text(md, encoding="utf-8")
    return decision


def _decision_md(decision, strict_row, grp_strict, full_row, audit, metrics_dir) -> str:
    def fmt(r):
        if not r:
            return "NOT RUN"
        return f"Accuracy={float(r.get('accuracy', float('nan'))):.4f}, F1_phishing={float(r.get('f1_phishing', float('nan'))):.4f} ({r.get('feature_set')}/{r.get('model')}) [see {metrics_dir.name}/{'final_random_test.csv' if (metrics_dir/'final_random_test.csv').exists() else 'quick_validation.csv'}]"
    l1 = fmt(full_row)
    l2 = l1  # F1 included above
    s_str = fmt(strict_row)
    g_str = fmt(grp_strict) if grp_strict else "NOT RUN"
    lines = []
    lines.append("# PhiUSIIL Decision — GO / NO-GO")
    lines.append("")
    lines.append("Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.")
    lines.append("")
    lines.append(f"(1) Accuracy best (random)? {l1}")
    lines.append(f"(2) F1 phishing (random best)? {l2}")
    lines.append(f"(3) Strict URL-only? {s_str}")
    lines.append(f"(4) Domain-disjoint? {g_str}")
    lines.append(f"(5) GIỮ hay ĐỔI và tại sao? {decision['status']}: {decision['decision_reason']}")
    lines.append("")
    if audit.get("SEVERE_COLLECTION_BIAS"):
        lines.append("> **BENCHMARK_NOT_REAL_WORLD_VALIDATED** — severe collection bias detected (e.g. path/https/www split by sampling). Do not claim real-world performance even if accuracy ~100%.")
        lines.append("")
    lines.append(f"- Status: `{decision['status']}` | academic_viability={decision['academic_viability']} | real_world_validity={decision['real_world_validity']}")
    lines.append(f"- Recommendation: {decision['recommendation']}")
    lines.append(f"- Dataset hash: `{decision.get('dataset_hash','')}`")
    lines.append(f"- Single-feature proxy: flag={decision['single_feature_proxy']['flag']} top={decision['single_feature_proxy'].get('top_feature')} acc={decision['single_feature_proxy'].get('top_acc')}")
    lines.append("")
    lines.append("## Evidence files")
    lines.append("- outputs/metrics/quick_validation.csv, outputs/metrics/final_random_test.csv, outputs/metrics/final_group_test.csv, outputs/metrics/domain_gap.csv")
    lines.append("- outputs/audit/label_conditional_stats.csv, outputs/audit/single_feature_stumps.csv, outputs/audit/feature_provenance.csv")
    lines.append("- outputs/figures/random_vs_domain.png, outputs/figures/confusion_matrix_best_strict.png, outputs/figures/roc_pr_best_models.png")
    lines.append("")
    lines.append("## Limitations")
    for item in decision["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Next steps")
    for item in decision["next_steps"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
