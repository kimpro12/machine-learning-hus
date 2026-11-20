"""Data analysis of the 21 PhishVN URL features (fit on the train split only).

A1  EDA: per-feature statistics by label, skewness, zeros, IQR outliers, correlations.
A2  Feature ranking + selection: ANOVA F, mutual information, single-feature AUC,
    top-k curve, L1 (Lasso) regularisation path.
A3  PCA: effect of scaling, loadings, per-component AUC, 2-D projection.
A4  Clustering: K-Means (elbow/silhouette/DB/CH), GMM (BIC), cluster profiles,
    agreement with label vs with source (ARI/NMI).
A5  t-SNE visualisation.

Everything unsupervised is fitted on train; validation is used only to score the
supervised read-outs (single-feature AUC, top-k curve, L1 path). The test split is
never touched here. Report: outputs/DATA_ANALYSIS.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, f1_score,
    normalized_mutual_info_score, roc_auc_score, silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .experiments import SOURCE_FAMILY, url_text
from .features import PHISHVN_FEATURES as FEATS

SEED = 42
K_RANGE = range(2, 11)
SIL_SAMPLE = 5000      # silhouette is O(n^2): score it on a fixed random subsample
TSNE_SAMPLE = 4000
PROFILE_K = range(4, 9)  # k for the cluster profile: best silhouette among 4..8 (enough detail to interpret)
LABEL = {0: 'benign', 1: 'phishing'}


# ---------------------------------------------------------------- shared preprocessing
def prep_pipeline(log: bool = True, scale: bool = True) -> Pipeline:
    """median impute -> (log1p) -> (StandardScaler); fitted on train only."""
    steps = [('impute', SimpleImputer(strategy='median'))]
    if log:
        steps.append(('log1p', FunctionTransformer(np.log1p)))
    if scale:
        steps.append(('scale', StandardScaler()))
    return Pipeline(steps)


def _logreg(**kw) -> LogisticRegression:
    return LogisticRegression(C=10.0, max_iter=3000, **kw)


def _sample(n: int, k: int, seed: int = SEED) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n, size=min(k, n), replace=False))


def _val_scores(y, score) -> tuple[float, float]:
    """(macro-F1 at 0.5, ROC-AUC) on validation."""
    return (float(f1_score(y, (score >= 0.5).astype(int), average='macro', zero_division=0)),
            float(roc_auc_score(y, score)))


# ---------------------------------------------------------------- A1 EDA
def eda(tr: pd.DataFrame) -> pd.DataFrame:
    X, y = tr[FEATS], tr['label'].to_numpy()
    rows = []
    for c in FEATS:
        x = X[c].astype(float)
        b, p = x[y == 0], x[y == 1]
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        out = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean()
        pooled = np.sqrt((b.var() + p.var()) / 2)
        rows.append({'feature': c,
                     'mean_benign': b.mean(), 'mean_phishing': p.mean(),
                     'median_benign': b.median(), 'median_phishing': p.median(),
                     'effect_size_d': (p.mean() - b.mean()) / pooled if pooled > 0 else 0.0,
                     'skew': x.skew(), 'pct_zero': (x == 0).mean(), 'pct_outlier_iqr': out,
                     'max': x.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- A2 ranking + selection
def feature_ranking(tr: pd.DataFrame, va: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ytr, yva = tr['label'].to_numpy(), va['label'].to_numpy()
    pp = prep_pipeline().fit(tr[FEATS])
    Ztr, Zva = pp.transform(tr[FEATS]), pp.transform(va[FEATS])
    with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F, _ = f_classif(Ztr, ytr)
    F = np.nan_to_num(F)  # constant feature (e.g. is_ip is always 0) -> F undefined -> 0
    mi = mutual_info_classif(Ztr, ytr, random_state=SEED)
    single = []
    for j in range(len(FEATS)):
        m = _logreg().fit(Ztr[:, [j]], ytr)
        auc = roc_auc_score(yva, m.predict_proba(Zva[:, [j]])[:, 1])
        single.append(max(auc, 1 - auc))  # direction-free: 0.5 = useless, 1 = perfect
    rank = pd.DataFrame({'feature': FEATS, 'anova_F': F, 'mutual_info': mi, 'single_feature_val_auc': single})
    rank = rank.sort_values('mutual_info', ascending=False).reset_index(drop=True)
    rank['mi_rank'] = np.arange(1, len(rank) + 1)
    rank['F_rank'] = rank['anova_F'].rank(ascending=False).astype(int)
    rank['auc_rank'] = rank['single_feature_val_auc'].rank(ascending=False).astype(int)

    # top-k curve (features ordered by train mutual information)
    curve = []
    for k in range(1, len(FEATS) + 1):
        cols = [FEATS.index(f) for f in rank['feature'][:k]]
        m = _logreg().fit(Ztr[:, cols], ytr)
        mf1, auc = _val_scores(yva, m.predict_proba(Zva[:, cols])[:, 1])
        curve.append({'k': k, 'added_feature': rank['feature'][k - 1], 'val_macro_f1': mf1, 'val_auc': auc})
    curve = pd.DataFrame(curve)

    # L1 regularisation path: smaller C -> stronger penalty -> more weights exactly 0
    path, entry = [], {}
    for C in np.logspace(-3, 1, 13):
        m = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=5000).fit(Ztr, ytr)
        nz = np.flatnonzero(np.abs(m.coef_[0]) > 1e-8)
        for j in nz:
            entry.setdefault(FEATS[j], C)
        mf1, auc = _val_scores(yva, m.predict_proba(Zva)[:, 1])
        path.append({'C': C, 'n_nonzero': len(nz), 'val_macro_f1': mf1, 'val_auc': auc,
                     'kept': ' '.join(FEATS[j] for j in nz)})
    rank['l1_entry_C'] = rank['feature'].map(entry)
    return rank, curve, pd.DataFrame(path)


# ---------------------------------------------------------------- A3 PCA
def pca_analysis(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    ytr, yva = tr['label'].to_numpy(), va['label'].to_numpy()
    comp = []
    for name, log, scale in [('raw (no scaling)', False, False), ('StandardScaler', False, True),
                             ('log1p + StandardScaler', True, True)]:
        Z = prep_pipeline(log, scale).fit_transform(tr[FEATS])
        p = PCA(random_state=SEED).fit(Z)
        cum = np.cumsum(p.explained_variance_ratio_)
        top = np.argsort(-np.abs(p.components_[0]))[:3]
        comp.append({'preprocessing': name, 'pc1_var': p.explained_variance_ratio_[0],
                     'pc2_var': p.explained_variance_ratio_[1],
                     'n_for_90pct': int(np.searchsorted(cum, 0.90) + 1),
                     'n_for_95pct': int(np.searchsorted(cum, 0.95) + 1),
                     'pc1_top_features': ', '.join(f"{FEATS[j]} ({p.components_[0][j]:+.2f})" for j in top)})
    pp = prep_pipeline().fit(tr[FEATS])
    Ztr, Zva = pp.transform(tr[FEATS]), pp.transform(va[FEATS])
    p = PCA(random_state=SEED).fit(Ztr)
    loadings = pd.DataFrame(p.components_[:5].T, index=FEATS, columns=[f'PC{i}' for i in range(1, 6)])
    Ptr, Pva = p.transform(Ztr), p.transform(Zva)
    pc_auc = []
    for i in range(10):
        m = _logreg().fit(Ptr[:, [i]], ytr)
        auc = roc_auc_score(yva, m.predict_proba(Pva[:, [i]])[:, 1])
        pc_auc.append({'component': f'PC{i + 1}', 'var_ratio': p.explained_variance_ratio_[i],
                       'single_pc_val_auc': max(auc, 1 - auc)})
    return {'scaling': pd.DataFrame(comp), 'loadings': loadings, 'pc_auc': pd.DataFrame(pc_auc),
            'proj': Ptr[:, :2]}


# ---------------------------------------------------------------- A4 clustering
def _profile_text(row: pd.Series, overall: pd.Series) -> str:
    """Plain-language label for a cluster from its medians/rates vs the whole train set."""
    bits = []
    bits.append('có `www.`' if row['www_rate'] > 0.5 else ('không `www.`' if row['www_rate'] < 0.1 else ''))
    bits.append('đuôi .vn' if row['vn_rate'] > 0.5 else ('không phải .vn' if row['vn_rate'] < 0.1 else ''))
    bits.append('có đường dẫn' if row['median_path_len'] > 0 else 'chỉ domain')
    if row['median_query_len'] > 0:
        bits.append('có tham số truy vấn')
    if row['median_digit_cnt'] > overall['digit_cnt_q75']:
        bits.append('nhiều chữ số')
    if row['median_dash_cnt'] > overall['dash_cnt_q75']:
        bits.append('nhiều dấu gạch ngang')
    if row['median_url_len'] > overall['url_len_q75']:
        bits.append('URL dài')
    elif row['median_url_len'] < overall['url_len_q25']:
        bits.append('URL ngắn')
    return ', '.join(b for b in bits if b)


def clustering(tr: pd.DataFrame) -> dict:
    Z = prep_pipeline().fit_transform(tr[FEATS])
    y = tr['label'].to_numpy()
    fam = tr['source'].map(SOURCE_FAMILY).to_numpy()
    si = _sample(len(Z), SIL_SAMPLE)
    rows, km_labels = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        lab = km.labels_
        km_labels[k] = lab
        rows.append({'method': 'kmeans', 'k': k, 'inertia': km.inertia_,
                     'silhouette': silhouette_score(Z[si], lab[si]),
                     'davies_bouldin': davies_bouldin_score(Z, lab),
                     'calinski_harabasz': calinski_harabasz_score(Z, lab),
                     'ari_label': adjusted_rand_score(y, lab), 'ari_source': adjusted_rand_score(fam, lab),
                     'nmi_label': normalized_mutual_info_score(y, lab), 'nmi_source': normalized_mutual_info_score(fam, lab)})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gm = GaussianMixture(n_components=k, covariance_type='full', random_state=SEED, reg_covar=1e-4).fit(Z)
        gl = gm.predict(Z)
        rows.append({'method': 'gmm_full', 'k': k, 'bic': gm.bic(Z), 'aic': gm.aic(Z),
                     'silhouette': silhouette_score(Z[si], gl[si]) if len(np.unique(gl[si])) > 1 else np.nan,
                     'ari_label': adjusted_rand_score(y, gl), 'ari_source': adjusted_rand_score(fam, gl),
                     'nmi_label': normalized_mutual_info_score(y, gl), 'nmi_source': normalized_mutual_info_score(fam, gl)})
    sel = pd.DataFrame(rows)

    # profile: best silhouette among PROFILE_K
    km_sel = sel[(sel['method'] == 'kmeans') & sel['k'].isin(list(PROFILE_K))]
    kp = int(km_sel.loc[km_sel['silhouette'].idxmax(), 'k'])
    lab = km_labels[kp]
    www = url_text(tr['url']).str.startswith('www.').to_numpy()
    vn = tr['is_vn'].to_numpy()
    overall = pd.Series({f'{c}_q{q}': np.percentile(tr[c], q) for c in ('digit_cnt', 'dash_cnt', 'url_len') for q in (25, 75)})
    prof = []
    for c in range(kp):
        mk = lab == c
        fams = pd.Series(fam[mk]).value_counts(normalize=True)
        r = {'cluster': c, 'n': int(mk.sum()), 'share': mk.mean(), 'pct_phishing': y[mk].mean(),
             'top_source': fams.index[0], 'top_source_share': fams.iloc[0],
             'www_rate': www[mk].mean(), 'vn_rate': vn[mk].mean()}
        for f in ('url_len', 'dom_len', 'subdom_cnt', 'tld_len', 'digit_cnt', 'dash_cnt', 'path_len', 'query_len', 'entropy'):
            r[f'median_{f}'] = float(np.median(tr[f].to_numpy()[mk]))
        r['description'] = _profile_text(pd.Series(r), overall)
        prof.append(r)
    prof = pd.DataFrame(prof).sort_values('n', ascending=False).reset_index(drop=True)

    return {'selection': sel, 'profile_k': kp, 'profiles': prof}


# ---------------------------------------------------------------- A5 t-SNE
def tsne_embed(tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = prep_pipeline().fit_transform(tr[FEATS])
    ti = _sample(len(Z), TSNE_SAMPLE, SEED + 2)
    E = TSNE(n_components=2, perplexity=30, init='pca', random_state=SEED).fit_transform(Z[ti])
    return E, tr['label'].to_numpy()[ti], tr['source'].map(SOURCE_FAMILY).to_numpy()[ti]


# ---------------------------------------------------------------- figures
FAM_ORDER = ['chongluadao', 'openphish', 'tinnhiemmang', 'tinnhiem (org+web)', 'tranco (global+vn)']
FAM_COLOR = {'chongluadao': '#d62728', 'openphish': '#ff7f0e', 'tinnhiemmang': '#8c564b',
             'tinnhiem (org+web)': '#1f77b4', 'tranco (global+vn)': '#2ca02c'}
LAB_COLOR = {0: '#1f77b4', 1: '#d62728'}


def _scatter_pair(P: np.ndarray, y: np.ndarray, fam: np.ndarray, title: str, path: Path, xl: str, yl: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for lab in (0, 1):
        m = y == lab
        axes[0].scatter(P[m, 0], P[m, 1], s=3, alpha=0.35, c=LAB_COLOR[lab], label=LABEL[lab])
    axes[0].set_title('colour = label')
    for f in FAM_ORDER:
        m = fam == f
        axes[1].scatter(P[m, 0], P[m, 1], s=3, alpha=0.35, c=FAM_COLOR[f], label=f)
    axes[1].set_title('colour = source family')
    for ax in axes:
        ax.set_xlabel(xl)
        ax.legend(markerscale=4, fontsize=7, loc='best')
    axes[0].set_ylabel(yl)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(tr, stats, rank, curve, l1, pca, clu, tsne, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = tr['label'].to_numpy()

    # A1 distributions (log1p scale) by label
    fig, axes = plt.subplots(3, 7, figsize=(18, 7.5))
    for ax, c in zip(axes.ravel(), FEATS):
        x = np.log1p(tr[c].to_numpy(dtype=float))
        bins = np.linspace(x.min(), x.max() if x.max() > x.min() else x.min() + 1, 30)
        for lab in (0, 1):
            ax.hist(x[y == lab], bins=bins, alpha=0.5, density=True, color=LAB_COLOR[lab], label=LABEL[lab])
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=6)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle('Feature distributions by label (x = log1p(value), density, train)')
    fig.tight_layout()
    fig.savefig(fig_dir / 'eda_distributions.png', dpi=130)
    plt.close(fig)

    # A1 correlation heatmap
    r = tr[FEATS].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(r, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATS)))
    ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels(FEATS, fontsize=7)
    for i in range(len(FEATS)):
        for j in range(len(FEATS)):
            if abs(r[i, j]) >= 0.7 and i != j:
                ax.text(j, i, f'{r[i, j]:.2f}', ha='center', va='center', fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title('Feature correlation (train); |r| >= 0.7 annotated')
    fig.tight_layout()
    fig.savefig(fig_dir / 'eda_correlation.png', dpi=150)
    plt.close(fig)

    # A2 ranking + top-k curve + L1 path
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    rk = rank.sort_values('mutual_info')
    axes[0].barh(rk['feature'], rk['mutual_info'], color='#4c72b0')
    axes[0].set_xlabel('mutual information with label (train)')
    axes[0].set_title('Feature ranking')
    axes[0].tick_params(axis='y', labelsize=7)
    axes[1].plot(curve['k'], curve['val_macro_f1'], marker='o', label='val macro-F1')
    axes[1].plot(curve['k'], curve['val_auc'], marker='s', label='val ROC-AUC')
    axes[1].set_xlabel('number of top-MI features kept')
    axes[1].set_title('Performance vs number of features (LogReg)')
    axes[1].legend()
    axes[2].semilogx(l1['C'], l1['val_macro_f1'], marker='o', label='val macro-F1')
    ax2 = axes[2].twinx()
    ax2.semilogx(l1['C'], l1['n_nonzero'], marker='s', color='#dd8452', label='non-zero weights')
    ax2.set_ylabel('features kept (non-zero weights)')
    axes[2].set_xlabel('C (smaller = stronger L1 penalty)')
    axes[2].set_ylabel('val macro-F1')
    axes[2].set_title('L1 (Lasso) regularisation path')
    fig.tight_layout()
    fig.savefig(fig_dir / 'feature_selection.png', dpi=150)
    plt.close(fig)

    # A3 PCA loadings + per-component AUC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.1, 1]})
    L = pca['loadings']
    im = axes[0].imshow(L.to_numpy(), cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    axes[0].set_yticks(range(len(L)))
    axes[0].set_yticklabels(L.index, fontsize=7)
    axes[0].set_xticks(range(L.shape[1]))
    axes[0].set_xticklabels(L.columns)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label='loading')
    axes[0].set_title('PCA loadings (log1p + StandardScaler, train)')
    pa = pca['pc_auc']
    x = np.arange(len(pa))
    axes[1].bar(x - 0.2, pa['var_ratio'], 0.4, label='share of variance')
    axes[1].bar(x + 0.2, pa['single_pc_val_auc'] - 0.5, 0.4, label='val AUC − 0.5 using this PC alone')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pa['component'])
    axes[1].set_title('Variance vs usefulness for the label')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'pca_loadings.png', dpi=150)
    plt.close(fig)
    pi = _sample(len(tr), 8000, SEED + 3)
    _scatter_pair(pca['proj'][pi], y[pi], tr['source'].map(SOURCE_FAMILY).to_numpy()[pi],
                  'PCA projection of train URLs (8,000 sampled)', fig_dir / 'pca_2d_label_source.png', 'PC1', 'PC2')

    # A4 k selection
    sel = clu['selection']
    km, gm = sel[sel['method'] == 'kmeans'], sel[sel['method'] == 'gmm_full']
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    axes[0].plot(km['k'], km['inertia'], marker='o')
    axes[0].set_title('K-Means elbow (inertia)')
    axes[1].plot(km['k'], km['silhouette'], marker='o', label='K-Means')
    axes[1].plot(gm['k'], gm['silhouette'], marker='s', label='GMM')
    axes[1].set_title('Silhouette (higher = better separated)')
    axes[1].legend(fontsize=8)
    axes[2].plot(gm['k'], gm['bic'], marker='o', color='#55a868')
    axes[2].set_title('GMM (full covariance) BIC (lower = better)')
    for d, nm, mk in ((km, 'K-Means', 'o'), (gm, 'GMM', 's')):
        axes[3].plot(d['k'], d['ari_label'], marker=mk, label=f'{nm}: ARI vs label')
        axes[3].plot(d['k'], d['ari_source'], marker=mk, linestyle='--', label=f'{nm}: ARI vs source')
    axes[3].set_title('Do clusters match label or source?')
    axes[3].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('k')
    fig.tight_layout()
    fig.savefig(fig_dir / 'clustering_selection.png', dpi=150)
    plt.close(fig)

    # A4 cluster profile heatmap (z-scored medians)
    pr = clu['profiles']
    cols = ['pct_phishing', 'www_rate', 'vn_rate'] + [c for c in pr.columns if c.startswith('median_')]
    M = pr[cols].to_numpy(dtype=float)
    Mz = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(pr) + 2))
    ax.imshow(Mz, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace('median_', '') for c in cols], rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(pr)))
    ax.set_yticklabels([f"cluster {int(r.cluster)} (n={int(r.n)}, {r.top_source})" for r in pr.itertuples()], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f'{v:.2f}' if cols[j].endswith('rate') or cols[j] == 'pct_phishing' or cols[j] == 'median_entropy' else f'{v:.0f}',
                    ha='center', va='center', fontsize=7)
    ax.set_title(f'K-Means k={clu["profile_k"]} cluster profiles (cell = value; colour = relative to other clusters)')
    fig.tight_layout()
    fig.savefig(fig_dir / 'cluster_profiles.png', dpi=150)
    plt.close(fig)

    # A5 t-SNE
    E, ty, tf = tsne
    _scatter_pair(E, ty, tf, f't-SNE of {len(E):,} sampled train URLs (visualisation only)',
                  fig_dir / 'tsne_label_source.png', 't-SNE 1', 't-SNE 2')


# ---------------------------------------------------------------- report
def _f(x, nd: int = 3) -> str:
    return 'n/a' if pd.isna(x) else f'{x:.{nd}f}'


def _table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(fmt[c](r[c]) if c in fmt else (_f(r[c]) if isinstance(r[c], float) else str(r[c]))
                                     for c in cols) + ' |')
    return '\n'.join(out)


def write_report(tr, va, stats, rank, curve, l1, pca, clu, path: Path) -> None:
    pct = lambda x: f'{x:.1%}'  # noqa: E731
    n_tr = len(tr)
    by_d = stats.reindex(stats['effect_size_d'].abs().sort_values(ascending=False).index)
    skewed = stats[stats['skew'] > 5]
    many0 = stats[(stats['pct_zero'] > 0.5) & (stats['pct_zero'] < 1)]
    const = [c for c in FEATS if tr[c].nunique() <= 1]
    corr = tr[FEATS].corr()
    hi = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATS) for b in FEATS[i + 1:] if abs(corr.loc[a, b]) > 0.9]

    best = curve['val_macro_f1'].max()
    k99 = int(curve.loc[curve['val_macro_f1'] >= best - 0.005, 'k'].min())
    jumps = curve['val_macro_f1'].diff()
    jk = int(curve.loc[jumps.idxmax(), 'k'])
    jfeat = curve.loc[jumps.idxmax(), 'added_feature']
    jsize = float(jumps.max())
    l1_best = l1['val_macro_f1'].max()
    l1_small = l1[l1['val_macro_f1'] >= l1_best - 0.005].sort_values('n_nonzero').iloc[0]
    top5 = ', '.join(f'`{f}`' for f in rank['feature'][:5])
    bottom = rank.sort_values('single_feature_val_auc').head(3)

    sc = pca['scaling']
    raw, ls = sc.iloc[0], sc.iloc[2]
    pa = pca['pc_auc']
    best_pc = pa.loc[pa['single_pc_val_auc'].idxmax()]
    L = pca['loadings']
    pc_desc = []
    for pc in ['PC1', 'PC2', 'PC3']:
        top = L[pc].abs().sort_values(ascending=False).index[:4]
        pc_desc.append(f"- **{pc}**: " + ', '.join(f"`{f}` ({L.loc[f, pc]:+.2f})" for f in top))

    sel = clu['selection']
    km = sel[sel['method'] == 'kmeans'].set_index('k')
    gm = sel[sel['method'] == 'gmm_full'].set_index('k')
    k_sil = int(km['silhouette'].idxmax())
    k_bic = int(gm['bic'].idxmin())
    src_wins = int((km['ari_source'] > km['ari_label']).sum())
    prof = clu['profiles']
    prof_lines = ['| Cụm | Số URL (tỷ lệ) | % phishing | Nguồn chính (tỷ lệ) | `www.` | .vn | Mô tả tự động |', '|---|---|---|---|---|---|---|']
    for r in prof.itertuples():
        prof_lines.append(f"| {r.cluster} | {r.n:,} ({pct(r.share)}) | {pct(r.pct_phishing)} | {r.top_source} ({pct(r.top_source_share)}) | "
                          f"{pct(r.www_rate)} | {pct(r.vn_rate)} | {r.description} |")
    big = prof[prof['share'] >= 0.01]
    tiny = prof[prof['share'] < 0.01]
    pure = big[(big['pct_phishing'] > 0.95) | (big['pct_phishing'] < 0.05)]
    pure_src = big[big['top_source_share'] > 0.8]
    top2 = prof.head(2)
    rate_cols = ['www_rate', 'vn_rate', 'pct_phishing']
    gap_col = max(rate_cols, key=lambda c: abs(top2[c].iloc[0] - top2[c].iloc[1]))
    nonlin = rank[((rank['F_rank'] - rank['mi_rank']) >= 8) & ~rank['feature'].isin(const)]

    txt = f"""# PhishVN — Phân tích dữ liệu (EDA, chọn feature, PCA, phân cụm)

Mọi phân tích dùng **tập train** ({n_tr:,} URL) của PhishVN. Tập val chỉ dùng để chấm các thí nghiệm có giám sát (AUC từng feature, đường cong top-k, đường L1). **Tập test không được dùng ở đây.**
Trừ khi ghi khác, dữ liệu được tiền xử lý bằng: điền median → `log1p` → `StandardScaler`. Lý do dùng `log1p` nằm ở mục A1.

---

## A1. Khám phá dữ liệu (EDA)

**Làm gì:** thống kê từng feature theo nhãn: trung bình, trung vị, độ lệch (skew), tỷ lệ bằng 0, tỷ lệ ngoại lai theo quy tắc IQR, và **effect size** (Cohen's d = chênh lệch trung bình hai lớp chia cho độ lệch chuẩn gộp; |d| ≥ 0.8 được coi là lớn).
**Tại sao:** trước khi train phải biết feature nào khác nhau giữa hai lớp, feature nào lệch hoặc có nhiều ngoại lai (ảnh hưởng tới cách chuẩn hóa), và feature nào trùng thông tin nhau.

{_table(by_d, ['feature', 'mean_benign', 'mean_phishing', 'median_benign', 'median_phishing', 'effect_size_d', 'skew', 'pct_zero', 'pct_outlier_iqr'],
        {'pct_zero': pct, 'pct_outlier_iqr': pct, 'effect_size_d': lambda x: f'{x:+.2f}', 'skew': lambda x: f'{x:.1f}'})}

(Sắp theo |effect size| giảm dần.)

**Đọc kết quả:**
- Feature phân biệt hai lớp rõ nhất (|d| lớn nhất): {', '.join(f"`{r.feature}` (d={r.effect_size_d:+.2f})" for r in by_d.head(4).itertuples())}. Dấu `+` nghĩa là phishing có giá trị lớn hơn.
- **{len(skewed)}/{len(FEATS)} feature lệch phải rất mạnh** (skew > 5), lớn nhất là {', '.join(f"`{r.feature}` ({r.skew:.0f})" for r in skewed.sort_values('skew', ascending=False).head(3).itertuples())}. **{len(many0)} feature có hơn 50% giá trị bằng 0** ({', '.join(f'`{f}`' for f in many0['feature'])}). Vì vậy trước khi dùng các phương pháp dựa trên khoảng cách (PCA, K-Means), dữ liệu được nén bằng `log1p(x) = log(1 + x)`: giá trị 0 giữ nguyên 0, giá trị cực lớn bị kéo lại gần.
- **Feature hằng số (vô dụng):** {', '.join(f'`{c}` (luôn = {tr[c].iloc[0]:g})' for c in const) or 'không có'}. Một feature không bao giờ thay đổi thì không mang thông tin gì; có thể bỏ mà không mất gì.
- **Cặp feature tương quan mạnh (|r| > 0.9):** {'; '.join(f'`{a}`–`{b}` ({r:+.2f})' for a, b, r in hi) or 'không có'}. Các cặp này gần như mang cùng một thông tin, nên giảm chiều hoặc lọc feature là hợp lý.

Hình: `figures/eda_distributions.png` (phân phối từng feature theo nhãn), `figures/eda_correlation.png` (ma trận tương quan).

---

## A2. Xếp hạng và chọn feature

**Làm gì:** chấm điểm mức hữu ích của từng feature theo ba cách:
- **ANOVA F:** trung bình của feature ở hai lớp có khác nhau nhiều so với độ dao động bên trong mỗi lớp không.
- **Mutual information (MI):** biết giá trị feature thì giảm được bao nhiêu độ bất định về nhãn. Khác ANOVA, MI bắt được cả quan hệ phi tuyến.
- **AUC khi chỉ dùng một feature:** train LogReg chỉ với feature đó, đo ROC-AUC trên val (0.5 = vô dụng, 1 = hoàn hảo).

Sau đó thử hai cách **chọn feature**: (1) giữ k feature có MI cao nhất, k = 1..21; (2) **L1 / Lasso**: phạt tổng |trọng số| khiến các feature yếu bị đưa trọng số về đúng 0.
**Tại sao:** ít feature hơn thì model đơn giản và dễ giải thích hơn, và cho biết thông tin thật sự nằm ở đâu.

{_table(rank, ['mi_rank', 'feature', 'mutual_info', 'anova_F', 'single_feature_val_auc', 'F_rank', 'auc_rank', 'l1_entry_C'],
        {'anova_F': lambda x: f'{x:,.0f}', 'l1_entry_C': lambda x: 'không vào' if pd.isna(x) else f'{x:.3g}', 'mutual_info': lambda x: f'{x:.3f}'})}

`l1_entry_C`: giá trị C nhỏ nhất (phạt mạnh nhất) mà feature vẫn có trọng số khác 0 trên đường L1. C càng nhỏ mà feature vẫn được giữ thì feature càng quan trọng.

**Đọc kết quả:**
- 5 feature có MI cao nhất: {top5}.{(' **Quan hệ phi tuyến:** ' + ', '.join(f'`{r.feature}` (MI hạng {r.mi_rank}, ANOVA F hạng {r.F_rank})' for r in nonlin.itertuples()) + ' được MI đánh giá cao hơn hẳn ANOVA. ANOVA chỉ so trung bình hai lớp, nên bỏ sót feature mà hai lớp có trung bình gần nhau nhưng phân phối khác nhau.') if len(nonlin) else ''} Feature gần như vô dụng khi đứng một mình (AUC thấp nhất): {', '.join(f"`{r.feature}` ({r.single_feature_val_auc:.3f})" for r in bottom.itertuples())}.
- **Đường cong top-k:** dùng đủ 21 feature đạt val macro-F1 {_f(curve['val_macro_f1'].iloc[-1])}; tốt nhất {_f(best)} tại k={int(curve.loc[curve['val_macro_f1'].idxmax(), 'k'])}. Chỉ cần **{k99} feature** đầu là đã cách mức tốt nhất không quá 0.005. Bước nhảy lớn nhất (+{jsize:.3f} macro-F1) xảy ra ở k={jk}, khi thêm **`{jfeat}`**.{' Đây chính là feature chứa tiền tố `www.` (www được tính là một subdomain), tức dấu vết thu thập dữ liệu đã nói ở `EXPERIMENTS.md` mục 0: phần lớn khả năng phân loại của 21 feature số đến từ nó.' if jfeat == 'subdom_cnt' else ''}
- **Đường L1:** với C={l1_small['C']:.3g}, model chỉ giữ **{int(l1_small['n_nonzero'])} feature** mà val macro-F1 = {_f(l1_small['val_macro_f1'])}, cách mức tốt nhất của đường L1 ({_f(l1_best)}) không quá 0.005. Các feature được giữ: {', '.join(f'`{f}`' for f in l1_small['kept'].split())}.

Hình: `figures/feature_selection.png`. Dữ liệu: `feature_ranking.csv`, `feature_selection_curve.csv`, `l1_path.csv`.

---

## A3. PCA (phân tích thành phần chính)

**Làm gì:** PCA tìm các hướng (thành phần chính, PC) mà dữ liệu biến thiên nhiều nhất. Mỗi PC là tổ hợp tuyến tính của 21 feature; trọng số của từng feature trong một PC gọi là **loading**.
**Tại sao:** giảm chiều, khử các feature trùng thông tin, và vẽ dữ liệu 21 chiều lên mặt phẳng 2D.

### A3.1 Vì sao phải chuẩn hóa trước khi làm PCA
{_table(sc, ['preprocessing', 'pc1_var', 'pc2_var', 'n_for_90pct', 'n_for_95pct', 'pc1_top_features'], {'pc1_var': pct, 'pc2_var': pct})}

Không chuẩn hóa thì PC1 chiếm {pct(raw['pc1_var'])} phương sai và bị chi phối bởi {raw['pc1_top_features'].split(',')[0]}: PCA chỉ "thấy" feature có đơn vị lớn nhất. Sau `log1p + StandardScaler`, phương sai trải đều hơn (PC1 {pct(ls['pc1_var'])}) và cần {int(ls['n_for_90pct'])} thành phần để giữ 90% thông tin. **Đây là lý do luôn chuẩn hóa trước PCA.**

### A3.2 Các thành phần chính có ý nghĩa gì (loadings)
{chr(10).join(pc_desc)}

### A3.3 Phương sai lớn có đồng nghĩa với hữu ích cho việc phân loại không?
{_table(pa, ['component', 'var_ratio', 'single_pc_val_auc'], {'var_ratio': pct})}

PC phân biệt phishing tốt nhất khi đứng một mình là **{best_pc['component']}** (AUC {_f(best_pc['single_pc_val_auc'])}), dù nó chỉ chiếm {pct(best_pc['var_ratio'])} phương sai. PCA chọn hướng theo **phương sai**, không theo **nhãn**; hướng biến thiên nhiều nhất không nhất thiết là hướng tách phishing khỏi benign. Điều này giải thích vì sao giảm chiều bằng PCA làm accuracy giảm (RQ3 trong `QUICK_RESULTS.md`).

Hình: `figures/pca_loadings.png`, `figures/pca_2d_label_source.png` (chiếu lên PC1–PC2, tô màu theo nhãn và theo nguồn). Dữ liệu: `pca_scaling_comparison.csv`, `pca_loadings.csv`, `pca_component_auc.csv`.

---

## A4. Phân cụm (K-Means, GMM)

**Làm gì:** gom URL thành k nhóm mà **không dùng nhãn**, với k = 2..10, bằng hai thuật toán:
- **K-Means:** mỗi cụm có một tâm; mỗi URL thuộc về tâm gần nhất. Lặp lại cho tới khi tâm không đổi.
- **GMM (Gaussian Mixture Model):** mỗi cụm là một phân phối chuẩn nhiều chiều (hình elip, ma trận hiệp phương sai đầy đủ); mỗi URL có xác suất thuộc từng cụm.

Cách chọn k (không cần nhãn):
- **Elbow:** inertia (tổng bình phương khoảng cách tới tâm) giảm khi k tăng; chọn chỗ đường cong "gãy khúc".
- **Silhouette** (−1..1, càng cao càng tốt): mỗi điểm gần cụm của mình hơn cụm khác bao nhiêu. Tính trên {SIL_SAMPLE:,} URL lấy mẫu.
- **Davies–Bouldin** (càng thấp càng tốt), **Calinski–Harabasz** (càng cao càng tốt).
- **BIC** cho GMM (càng thấp càng tốt): cân bằng giữa độ khớp dữ liệu và số tham số.

Cách đối chiếu với nhãn (chỉ để đánh giá, không dùng khi phân cụm): **ARI** (Adjusted Rand Index, 0 = như ngẫu nhiên, 1 = khớp hoàn toàn) và **NMI** (Normalized Mutual Information, 0..1). Đối chiếu với **cả nhãn lẫn họ nguồn** dữ liệu.

### A4.1 Chọn k
{_table(km.reset_index(), ['k', 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'ari_label', 'ari_source', 'nmi_label', 'nmi_source'],
        {'inertia': lambda x: f'{x:,.0f}', 'calinski_harabasz': lambda x: f'{x:,.0f}'})}

GMM (ma trận hiệp phương sai đầy đủ):

{_table(gm.reset_index(), ['k', 'bic', 'silhouette', 'ari_label', 'ari_source'], {'bic': lambda x: f'{x:,.0f}'})}

- Silhouette cao nhất ở **k={k_sil}** ({_f(km.loc[k_sil, 'silhouette'])}); GMM có BIC thấp nhất ở **k={k_bic}**.
- **Cụm khớp với nguồn nhiều hơn khớp với nhãn ở {src_wins}/{len(km)} giá trị k** (ARI vs nguồn so với ARI vs nhãn, K-Means). Cấu trúc tự nhiên của dữ liệu chủ yếu phản ánh **URL đến từ danh sách nào**. Đây là cùng hiện tượng phụ thuộc nguồn đo được ở `EXPERIMENTS.md`.

### A4.2 Hồ sơ từng cụm (K-Means, k={clu['profile_k']})
k được chọn là k có silhouette cao nhất trong khoảng {PROFILE_K.start}..{PROFILE_K.stop - 1}, đủ chi tiết để diễn giải. Cột "Mô tả tự động" được sinh từ trung vị và tỷ lệ của cụm so với toàn bộ train.

{chr(10).join(prof_lines)}

- Hai cụm lớn nhất (cụm {int(top2['cluster'].iloc[0])} và {int(top2['cluster'].iloc[1])}, cộng lại {pct(top2['share'].sum())} dữ liệu) khác nhau rõ nhất ở **{ {'www_rate': 'tỷ lệ `www.`', 'vn_rate': 'tỷ lệ đuôi .vn', 'pct_phishing': 'tỷ lệ phishing'}[gap_col] }** ({pct(top2[gap_col].iloc[0])} so với {pct(top2[gap_col].iloc[1])}). Khi không có nhãn, K-Means vẫn tự tìm ra đúng dấu vết thu thập mà `EXPERIMENTS.md` mục 0 đã chỉ ra.
- Trong {len(big)} cụm chiếm ≥ 1% dữ liệu: **{len(pure)} cụm gần như thuần một nhãn** (>95% hoặc <5% phishing), **{len(pure_src)} cụm có hơn 80% URL từ cùng một họ nguồn**.{f' Ngoài ra có {len(tiny)} cụm tí hon ({int(tiny[chr(110)].sum())} URL, chủ yếu từ ' + tiny['top_source'].mode().iloc[0] + '): K-Means tách riêng các URL ngoại lai này; xem cột mô tả trong bảng.' if len(tiny) else ''}
- Trung vị chi tiết từng feature theo cụm: `cluster_profiles.csv`.

Hình: `figures/clustering_selection.png`, `figures/cluster_profiles.png`. Dữ liệu: `clustering_selection.csv`, `cluster_profiles.csv`.

---

## A5. t-SNE (trực quan hóa)

**Làm gì:** t-SNE chiếu dữ liệu nhiều chiều xuống 2D sao cho các điểm **gần nhau** ở không gian gốc vẫn gần nhau trên hình. Chạy trên {TSNE_SAMPLE:,} URL lấy mẫu (perplexity 30).
**Tại sao:** PCA chỉ giữ cấu trúc tuyến tính; t-SNE cho thấy các cụm cục bộ phi tuyến.
**Lưu ý:** khoảng cách giữa các cụm trên hình t-SNE và kích thước cụm **không có ý nghĩa định lượng**; t-SNE chỉ dùng để nhìn, không dùng làm feature.

Hình: `figures/tsne_label_source.png`. So sánh hai bảng màu (nhãn và nguồn): nếu các đảo trên hình tách theo màu nguồn rõ hơn theo màu nhãn, dữ liệu đang phân nhóm theo nguồn.

---

## Liên hệ với thí nghiệm tiền xử lý
Ảnh hưởng thật của từng kỹ thuật (cắt ngoại lai, rời rạc hóa, feature đa thức, lọc tương quan, SelectKBest, L1, PCA, under/oversampling, cụm K-Means làm feature) lên Logistic Regression, đo trên val/test/Hassan kèm kiểm định thống kê, nằm ở `EXPERIMENTS.md` mục 1d.
"""
    path.write_text(txt, encoding='utf-8')


# ---------------------------------------------------------------- entry
def run_analysis(vv: pd.DataFrame, out_dir: Path, fig_dir: Path) -> None:
    tr = vv[vv['split'] == 'train'].reset_index(drop=True)
    va = vv[vv['split'] == 'val'].reset_index(drop=True)
    print('  [analysis] A1 EDA ...', flush=True)
    stats = eda(tr)
    stats.to_csv(out_dir / 'eda_feature_stats.csv', index=False)
    print('  [analysis] A2 feature ranking + selection ...', flush=True)
    rank, curve, l1 = feature_ranking(tr, va)
    rank.to_csv(out_dir / 'feature_ranking.csv', index=False)
    curve.to_csv(out_dir / 'feature_selection_curve.csv', index=False)
    l1.to_csv(out_dir / 'l1_path.csv', index=False)
    print('  [analysis] A3 PCA ...', flush=True)
    pca = pca_analysis(tr, va)
    pca['scaling'].to_csv(out_dir / 'pca_scaling_comparison.csv', index=False)
    pca['loadings'].to_csv(out_dir / 'pca_loadings.csv')
    pca['pc_auc'].to_csv(out_dir / 'pca_component_auc.csv', index=False)
    print('  [analysis] A4 clustering (K-Means, GMM) ...', flush=True)
    clu = clustering(tr)
    clu['selection'].to_csv(out_dir / 'clustering_selection.csv', index=False)
    clu['profiles'].to_csv(out_dir / 'cluster_profiles.csv', index=False)
    print('  [analysis] A5 t-SNE ...', flush=True)
    tsne = tsne_embed(tr)
    plot_all(tr, stats, rank, curve, l1, pca, clu, tsne, fig_dir)
    write_report(tr, va, stats, rank, curve, l1, pca, clu, out_dir / 'DATA_ANALYSIS.md')
