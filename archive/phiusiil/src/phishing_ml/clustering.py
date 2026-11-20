"""Clustering: K-Means + GMM on clean numeric + PCA (E5)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def run_clustering(df, feature_sets: dict, splits_random: dict, cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cl_cfg = cfg.get("clustering", {})
    max_sample = int(cl_cfg.get("max_sample", 25000))
    k_range = cl_cfg.get("k_range", [2, 3, 4, 5, 6])
    sil_sample = int(cl_cfg.get("silhouette_sample", 5000))
    # clean numeric: prefer NO_SUSPECT_DERIVED
    fs = "NO_SUSPECT_DERIVED" if "NO_SUSPECT_DERIVED" in feature_sets else list(feature_sets.keys())[0]
    feats = [f for f in feature_sets[fs] if f in df.columns]
    tr = np.array(splits_random["train"])
    rng = np.random.default_rng(42)
    if len(tr) > max_sample:
        tr_sub = rng.choice(tr, size=max_sample, replace=False)
    else:
        tr_sub = tr
    X = df.iloc[tr_sub][feats].copy()
    y = df.iloc[tr_sub]["y"].to_numpy()
    # fit impute/scale/PCA on this subsample (documented as train-subsample fit)
    imp = SimpleImputer(strategy="median")
    Xs = StandardScaler().fit_transform(imp.fit_transform(X))
    n_pca = min(10, Xs.shape[1], len(Xs) - 1)
    n_pca = max(2, n_pca)
    pca = PCA(n_components=n_pca, random_state=42)
    Z = pca.fit_transform(Xs)
    # 2D for viz
    Z2 = Z[:, :2]
    rows = []
    labels_store = {}
    for k in k_range:
        k = int(k)
        # KMeans
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lab = km.fit_predict(Z)
            sil = float(silhouette_score(Z[np.random.default_rng(0).choice(len(Z), size=min(sil_sample, len(Z)), replace=False)], lab[np.random.default_rng(0).choice(len(Z), size=min(sil_sample, len(Z)), replace=False)])) if k > 1 and len(Z) > k else float("nan")
            # simpler: silhouette on sample
            idx = np.random.default_rng(1).choice(len(Z), size=min(sil_sample, len(Z)), replace=False)
            sil = float(silhouette_score(Z[idx], lab[idx]))
            db = float(davies_bouldin_score(Z, lab))
            ch = float(calinski_harabasz_score(Z, lab))
            ari = float(adjusted_rand_score(y, lab))
            rows.append({"method": "kmeans", "k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "bic": None, "aic": None, "ari_vs_label": ari, "status": "OK"})
            labels_store[f"kmeans_{k}"] = lab
        except Exception as e:
            rows.append({"method": "kmeans", "k": k, "status": f"FAIL: {e}"})
        # GMM
        try:
            gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
            lab = gm.fit_predict(Z)
            idx = np.random.default_rng(1).choice(len(Z), size=min(sil_sample, len(Z)), replace=False)
            sil = float(silhouette_score(Z[idx], lab[idx])) if k > 1 else float("nan")
            db = float(davies_bouldin_score(Z, lab))
            ch = float(calinski_harabasz_score(Z, lab))
            ari = float(adjusted_rand_score(y, lab))
            rows.append({"method": "gmm_diag", "k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "bic": float(gm.bic(Z)), "aic": float(gm.aic(Z)), "ari_vs_label": ari, "status": "OK"})
            labels_store[f"gmm_{k}"] = lab
        except Exception as e:
            rows.append({"method": "gmm_diag", "k": k, "status": f"FAIL: {e}"})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out_dir / "clustering_metrics.csv", index=False)
    # cluster profile: phishing rate per cluster for k=2..3 best-effort
    try:
        prof_rows = []
        for key, lab in labels_store.items():
            for c in sorted(np.unique(lab).tolist()):
                m = lab == c
                prof_rows.append({"config": key, "cluster": int(c), "n": int(m.sum()), "phish_rate": float(y[m].mean()) if m.sum() else 0.0})
        pd.DataFrame(prof_rows).to_csv(out_dir / "cluster_class_composition.csv", index=False)
        # scatter 2D colored by cluster (kmeans k=2) + true label
        if "kmeans_2" in labels_store:
            lab = labels_store["kmeans_2"]
            plt.figure(figsize=(6, 5))
            plt.scatter(Z2[:, 0], Z2[:, 1], c=lab, s=3, alpha=0.5)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"KMeans k=2 on PCA(n={n_pca}) subsample n={len(Z2)} (ARI vs label posterior only)")
            plt.tight_layout()
            plt.savefig(fig_dir / "cluster_pca_scatter.png", dpi=150)
            plt.close()
        # composition bar
        prof = pd.DataFrame(prof_rows)
        sub = prof[prof["config"] == "kmeans_2"] if "kmeans_2" in prof["config"].unique().tolist() else prof.head(6)
        if len(sub):
            plt.figure(figsize=(7, 4))
            plt.bar(sub["cluster"].astype(str), sub["phish_rate"])
            plt.xlabel("cluster")
            plt.ylabel("phishing rate (y=1 share)")
            plt.title("Cluster class composition (posterior description only)")
            plt.tight_layout()
            plt.savefig(fig_dir / "cluster_class_composition.png", dpi=150)
            plt.close()
        # selection metrics
        ok = rdf[rdf["status"] == "OK"]
        if len(ok):
            plt.figure(figsize=(8, 4))
            for method in ok["method"].unique():
                s = ok[ok["method"] == method].sort_values("k")
                plt.plot(s["k"], s["silhouette"], marker="o", label=method)
            plt.xlabel("k")
            plt.ylabel("silhouette (sample)")
            plt.title("Cluster selection: silhouette vs k")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "cluster_selection_metrics.png", dpi=150)
            plt.close()
    except Exception:
        pass
    return rdf
