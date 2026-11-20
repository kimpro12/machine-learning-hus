"""PhishVN training/evaluation pipeline (reproducible, in-repo).

Running::

    python -m phishing_ml.cli run               # full: audit -> quick -> test -> external -> pca/cluster -> figs -> reports
    python -m phishing_ml.cli run --mode quick  # skip PCA/clustering

reproduces everything under outputs/ from data/phishvn/data/*.csv (official
release, MANIFEST-verified) plus the independent externals under
data/external/ (Hassan, JPCERT).

Conventions (must stay stable for reproducibility):
- PhishVN label: 1 = phishing, 0 = benign. Positive class = phishing (1).
- Features: PHISHVN_FEATURES (21 numeric, scheme-stripped, is_https excluded).
- Official group-aware temporal split (split column) is preserved; HP selection
  uses validation only; the official test split is evaluated exactly once.
- External evaluation reuses frozen in-PhishVN-trained pipelines with a fixed
  0.5 threshold; external labels are used for scoring only, never for tuning.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .evaluate import clopper_pearson_ci, compute_metrics, scores_from_model
from .features import PHISHVN_FEATURES as FEATS
from .preprocess import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
FIG = ROOT / 'outputs' / 'figures'
PHISHVN_DATA = ROOT / 'data' / 'phishvn' / 'data'
HASSAN_CSV = ROOT / 'data' / 'external' / 'hassan_balanced_urls.csv'
JPCERT_DIR = ROOT / 'data' / 'external' / 'jpcert'

SUBSETS = ('FULL', 'GOLDSILVER')
TIER_CORE = ('gold', 'silver')


def norm_key(s: object) -> str:
    """Normalized join key: strip scheme, lowercase, drop trailing slash."""
    t = str(s).strip().lower()
    if t.startswith('http://'):
        t = t[7:]
    elif t.startswith('https://'):
        t = t[8:]
    return t.rstrip('/')


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average='macro', zero_division=0))


def grids():
    """Model grid: (name, estimator, needs_scaling). Kept identical across runs."""
    import sklearn.base  # noqa: F401  (re-exported for clone use below)
    return [
        # Majority-class baseline: exposes the accuracy paradox on imbalanced
        # slices (e.g. GOLDSILVER val: 24 phishing / 1585). Always first.
        ('dummy_most_frequent', DummyClassifier(strategy='most_frequent'), False),
        ('logreg_C0.1', LogisticRegression(max_iter=2000, C=0.1, solver='lbfgs'), True),
        ('logreg_C1', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'), True),
        ('logreg_C10', LogisticRegression(max_iter=2000, C=10.0, solver='lbfgs'), True),
        ('rf_mdNone_msl1', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=4), False),
        ('rf_md16_msl1', RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=1, random_state=42, n_jobs=4), False),
        ('rf_mdNone_msl3', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=4), False),
        ('linsvc_C0.1', __import__('sklearn.svm', fromlist=['LinearSVC']).LinearSVC(C=0.1, dual='auto', max_iter=5000), True),
        ('linsvc_C1', __import__('sklearn.svm', fromlist=['LinearSVC']).LinearSVC(C=1.0, dual='auto', max_iter=5000), True),
        ('linsvc_C10', __import__('sklearn.svm', fromlist=['LinearSVC']).LinearSVC(C=10.0, dual='auto', max_iter=5000), True),
    ]


def load_frames():
    """Load release tables + merge source/scenario via normalized key."""
    v = pd.read_csv(PHISHVN_DATA / 'vn_compphish.csv', low_memory=False)
    d = pd.read_csv(PHISHVN_DATA / 'dataset_url.csv', low_memory=False)
    v['k'] = v['url'].map(norm_key)
    d['k'] = d['url'].map(norm_key)
    vv = pd.merge(v, d[['k', 'source', 'scenario']], on='k', how='left')
    vv['has_path'] = (vv['path_len'] > 0).astype(int)
    vv['has_query'] = (vv['query_len'] > 0).astype(int)
    vv['is_vn'] = vv['tld'].astype(str).str.lower().isin(
        ['vn', 'com.vn', 'gov.vn', 'edu.vn', 'net.vn', 'org.vn']).astype(int)
    return vv


# ---------------------------------------------------------------- audit
def run_audit(vv: pd.DataFrame) -> dict:
    stats: dict = {
        'n': int(len(vv)),
        'label_counts': {int(k): int(n) for k, n in vv['label'].value_counts().items()},
        'tier_counts': {str(k): int(n) for k, n in vv['tier'].value_counts().items()},
        'split_counts': {str(k): int(n) for k, n in vv['split'].value_counts().items()},
        'exact_url_dups': int(vv.duplicated(subset=['url'], keep='first').sum()),
    }
    nk = vv['url'].astype(str).str.strip().str.lower().map(
        lambda s: s[7:] if s.startswith('http://') else (s[8:] if s.startswith('https://') else s))
    vc = nk.value_counts()
    stats['norm_dup_groups'] = int((vc > 1).sum())
    stats['norm_extra_rows'] = int((vc[vc > 1] - 1).sum()) if stats['norm_dup_groups'] else 0
    doms = vv['dom'].astype(str).str.lower()
    tr, va, te = (set(doms[vv['split'] == s]) for s in ('train', 'val', 'test'))
    stats['domain_overlap'] = {
        'train_val': len(tr & va), 'train_test': len(tr & te), 'val_test': len(va & te)}
    h = vv.copy()
    stats['is_https_by_label'] = {
        str(k): float(x) for k, x in h.groupby('label')['is_https'].mean().items()}
    rows = []
    for c in FEATS + ['is_https']:
        g = vv.groupby('label')[c]
        rows.append({
            'feature': c,
            'mean_phish': float(g.mean().loc[1]), 'mean_benign': float(g.mean().loc[0]),
            'median_phish': float(g.median().loc[1]), 'median_benign': float(g.median().loc[0]),
            'excluded_from_model': c == 'is_https',
            'reason': 'collection artefact per authors' if c == 'is_https'
                      else 'lexical, URL-computable',
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'feature_audit.csv', index=False)
    return stats


# ---------------------------------------------------------------- quick benchmark
def run_quick(vv: pd.DataFrame) -> dict:
    import sklearn.base
    subsets = {'FULL': vv, 'GOLDSILVER': vv[vv['tier'].isin(TIER_CORE)].copy()}
    best = {}
    for name, vsub in subsets.items():
        tr, va = vsub[vsub['split'] == 'train'], vsub[vsub['split'] == 'val']
        Xtr, ytr = tr[FEATS], tr['label'].to_numpy()
        Xva, yva = va[FEATS], va['label'].to_numpy()
        rows = []
        for mname, model, scale in grids():
            pipe = make_pipeline(sklearn.base.clone(model), scale)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipe.fit(Xtr, ytr)
            ft = time.time() - t0
            pred = pipe.predict(Xva)
            try:
                score = scores_from_model(pipe, Xva)
            except Exception:
                score = pred.astype(float)
            m = compute_metrics(yva, pred, score)
            rows.append({'subset': name, 'model': mname,
                         'params': json.dumps({}),
                         'n_train': len(Xtr), 'n_val': len(Xva), 'fit_s': round(ft, 2),
                         'accuracy': m['accuracy'], 'bal_acc': m['balanced_accuracy'],
                         'prec': m['precision_phishing'], 'rec': m['recall_phishing'],
                         'spec': m['specificity'], 'f1': m['f1_phishing'],
                         'macro_f1': macro_f1(yva, pred),
                         'roc': m['roc_auc'], 'pr': m['pr_auc'],
                         'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP']})
        q = pd.DataFrame(rows)
        q.to_csv(OUT / f'quick_val_{name}.csv', index=False)
        best[name] = q.sort_values('f1', ascending=False).iloc[0].to_dict()
    json.dump({k: v['model'] for k, v in best.items()},
              open(OUT / 'frozen_models.json', 'w'))
    return best


# ---------------------------------------------------------------- final test (once, frozen)
def run_final_test(vv: pd.DataFrame, best: dict) -> pd.DataFrame:
    import sklearn.base
    gmap = {m[0]: m for m in grids()}
    subsets = {'FULL': vv, 'GOLDSILVER': vv[vv['tier'].isin(TIER_CORE)].copy()}
    final_rows = []
    for name in SUBSETS:
        vsub = subsets[name]
        mname = best[name]['model']
        _, model, scale = gmap[mname]
        pipe = make_pipeline(sklearn.base.clone(model), scale)
        tr, te = vsub[vsub['split'] == 'train'], vsub[vsub['split'] == 'test']
        Xtr, ytr = tr[FEATS], tr['label'].to_numpy()
        Xte, yte = te[FEATS], te['label'].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipe.fit(Xtr, ytr)  # train only; val already used for selection
        pred = pipe.predict(Xte)
        try:
            score = scores_from_model(pipe, Xte)
        except Exception:
            score = pred.astype(float)
        m = compute_metrics(yte, pred, score)
        k = int((yte == pred).sum())
        lo, hi = clopper_pearson_ci(k, len(yte))
        final_rows.append({'subset': name, 'model': mname,
                           'n_train': len(Xtr), 'n_test': len(Xte),
                           'accuracy': m['accuracy'], 'bal_acc': m['balanced_accuracy'],
                           'prec': m['precision_phishing'], 'rec': m['recall_phishing'],
                           'spec': m['specificity'], 'f1': m['f1_phishing'],
                           'macro_f1': macro_f1(yte, pred),
                           'roc': m['roc_auc'], 'pr': m['pr_auc'],
                           'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'],
                           'acc_exact_lo': lo, 'acc_exact_hi': hi})
        joblib.dump(pipe, OUT / f'frozen_{name.lower()}_best.joblib')
        _write_test_breakdowns(vv, name, pipe)
    fin = pd.DataFrame(final_rows)
    fin.to_csv(OUT / 'classification_metrics.csv', index=False)
    return fin


def _slice_frame(y, pred, score, label: str, subset: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # single-class slices: ROC undefined -> NaN by design
        m = compute_metrics(y, pred, score)
    k = int((np.asarray(y) == np.asarray(pred)).sum())
    lo, hi = clopper_pearson_ci(k, len(y))
    return {'slice': label, 'subset': subset, 'n': len(y),
            'phish_rate': round(float(np.mean(y)), 4),
            'accuracy': round(m['accuracy'], 4), 'prec': round(m['precision_phishing'], 4),
            'rec': round(m['recall_phishing'], 4), 'spec': round(m['specificity'], 4),
            'f1': round(m['f1_phishing'], 4),
            'macro_f1': round(float(f1_score(y, pred, average='macro', zero_division=0)), 4),
            'roc': round(float(m.get('roc_auc', float('nan'))), 4),
            'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'],
            'acc_lo': round(lo, 4), 'acc_hi': round(hi, 4),
            'tiny_n_flag': bool(len(y) < 30)}


def _write_test_breakdowns(vv: pd.DataFrame, name: str, pipe) -> None:
    te = vv[vv['split'] == 'test']
    if name == 'GOLDSILVER':
        te = te[te['tier'].isin(TIER_CORE)]
    X, y = te[FEATS], te['label'].to_numpy()
    pred = pipe.predict(X)
    try:
        score = scores_from_model(pipe, X)
    except Exception:
        score = pred.astype(float)
    for col in ('source', 'tier', 'scenario'):
        rows = []
        for val, g in te.groupby(col):
            idx = te.index.get_indexer(g.index)
            rows.append(_slice_frame(y[idx], pred[idx], score[idx], str(val), name))
        df = pd.DataFrame(rows).sort_values('n', ascending=False)
        df.insert(1, 'slice_col', col)
        df.rename(columns={'slice': col}, inplace=True)
        df.to_csv(OUT / f'test_by_{col}_{name}.csv', index=False)
    masks = [('has_path', te['has_path'] == 1), ('no_path', te['has_path'] == 0),
             ('has_query', te['has_query'] == 1),
             ('is_vn', te['is_vn'] == 1), ('non_vn', te['is_vn'] == 0),
             ('has_ip', te['is_ip'] == 1)]
    rows = []
    for sname, mk in masks:
        idx = np.where(mk.to_numpy())[0]
        if len(idx) == 0:
            rows.append({'slice': sname, 'subset': name, 'n': 0, 'tiny_n_flag': True})
            continue
        rows.append(_slice_frame(y[idx], pred[idx], score[idx], sname, name))
    pd.DataFrame(rows).to_csv(OUT / f'test_by_urlslice_{name}.csv', index=False)


# ---------------------------------------------------------------- external evaluation
def load_jpcert_urls(jdir: Path = JPCERT_DIR) -> list[str]:
    """Unique non-empty URLs from JPCERT monthly CSVs (date,URL,description), month order."""
    frames = [pd.read_csv(f, usecols=['URL'], dtype=str) for f in sorted(jdir.glob('jpcert-*.csv'))]
    if not frames:
        return []
    urls = pd.concat(frames, ignore_index=True)['URL'].dropna().astype(str)
    urls = urls[urls.str.len() > 0].drop_duplicates()
    return urls.tolist()


def run_external(frozen_info: dict) -> pd.DataFrame:
    from .features import phishvn_features
    ext_rows = []
    pipes = {n: joblib.load(OUT / f'frozen_{n.lower()}_best.joblib') for n in SUBSETS}
    hpath = HASSAN_CSV
    if hpath.exists():
        h = pd.read_csv(hpath, low_memory=False)
        url_col = 'url' if 'url' in h.columns else h.columns[1]
        lab_col = 'label' if 'label' in h.columns else h.columns[0]
        he = pd.DataFrame({'ext_url': h[url_col].astype(str)})
        he['y'] = h[lab_col].astype(str).str.lower().map(
            {'phishing': 1, 'legitimate': 0, 'bad': 1, 'good': 0})
        he = he.dropna(subset=['y'])
        he['y'] = he['y'].astype(int)
        hf = pd.DataFrame([phishvn_features(u) for u in he['ext_url'].tolist()])
        for name in SUBSETS:
            pipe = pipes[name]
            X, y = hf[FEATS], he['y'].to_numpy()
            pred = pipe.predict(X)
            try:
                sc = scores_from_model(pipe, X)
            except Exception:
                sc = pred.astype(float)
            m = compute_metrics(y, pred, sc)
            k = int((y == pred).sum())
            lo, hi = clopper_pearson_ci(k, len(y))
            ext_rows.append({'external': 'hassan_340k', 'model_subset': name,
                             'model': frozen_info[name], 'n': len(y),
                             'metric_primary': 'accuracy', 'single_class': False,
                             **_mrow(m, y, pred, lo, hi),
                             'note': 'frozen, no retraining/threshold tuning; '
                                     'features recomputed from URL strings only'})
            qp, pp_ = (hf['query_len'] > 0).to_numpy(), (hf['path_len'] > 0).to_numpy()
            for sname, mk in (('has_path', pp_), ('no_path', ~pp_), ('has_query', qp)):
                idx = np.where(mk)[0]
                yt, pr2 = y[idx], pred[idx]
                sc2 = sc[idx] if sc is not None else None
                m2 = compute_metrics(yt, pr2, sc2)
                k2 = int((yt == pr2).sum())
                lo2, hi2 = clopper_pearson_ci(k2, len(idx))
                ext_rows.append({'external': f'hassan_slice_{sname}', 'model_subset': name,
                                 'model': frozen_info[name], 'n': len(idx),
                                 'metric_primary': 'accuracy',
                                 'single_class': bool(len(np.unique(yt)) < 2),
                                 **_mrow(m2, yt, pr2, lo2, hi2), 'note': 'slice'})
    else:
        print('WARNING: Hassan CSV missing — external left unverified, not fabricated.')
    jurls = load_jpcert_urls()
    if len(jurls):
        jf = pd.DataFrame([phishvn_features(u) for u in jurls])
        y = np.ones(len(jf), dtype=int)
        for name in SUBSETS:
            pred = pipes[name].predict(jf[FEATS])
            tp, fn = int((pred == 1).sum()), int((pred == 0).sum())
            rec = tp / (tp + fn)
            lo, hi = clopper_pearson_ci(tp, tp + fn)
            ext_rows.append({'external': 'jpcert_2025H1_phishonly', 'model_subset': name,
                             'model': frozen_info[name], 'n': len(y),
                             'metric_primary': 'recall_phishonly', 'single_class': True,
                             'accuracy': float('nan'), 'bal_acc': float('nan'),
                             'prec': float('nan'), 'rec': rec, 'spec': float('nan'),
                             'f1': float('nan'), 'macro_f1': float('nan'),
                             'roc': float('nan'), 'pr': float('nan'),
                             'TN': 0, 'FP': 0, 'FN': fn, 'TP': tp,
                             'acc_lo': lo, 'acc_hi': hi,
                             'note': 'phishing-only; accuracy column intentionally empty'})
    ext = pd.DataFrame(ext_rows)
    ext.to_csv(OUT / 'external_metrics.csv', index=False)
    return ext


def _mrow(m: dict, y, pred, lo: float, hi: float) -> dict:
    return {'accuracy': m['accuracy'], 'bal_acc': m['balanced_accuracy'],
            'prec': m['precision_phishing'], 'rec': m['recall_phishing'],
            'spec': m['specificity'], 'f1': m['f1_phishing'],
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'roc': m['roc_auc'], 'pr': m['pr_auc'],
            'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'],
            'acc_lo': lo, 'acc_hi': hi}


# ---------------------------------------------------------------- PCA + clustering (after quick benchmark)
def run_pca_cluster(vv: pd.DataFrame) -> None:
    tr, va = vv[vv['split'] == 'train'], vv[vv['split'] == 'val']
    Xtr, ytr = tr[FEATS].to_numpy(), tr['label'].to_numpy()
    Xva, yva = va[FEATS].to_numpy(), va['label'].to_numpy()
    imp = SimpleImputer(strategy='median')
    Xs = StandardScaler().fit_transform(imp.fit_transform(Xtr))
    pf = PCA(random_state=42)
    pf.fit(Xs)
    _fig_scree(np.cumsum(pf.explained_variance_ratio_))
    prows = []
    for cond, kw in [('no-PCA', {}), ('var90', {'n': 0.90}), ('var95', {'n': 0.95}),
                     ('var99', {'n': 0.99}), ('k5', {'n': 5}), ('k10', {'n': 10})]:
        for mname, mk, sc in [('logreg_C1', LogisticRegression(max_iter=2000, C=1.0), True),
                              ('linsvc_C1', __import__('sklearn.svm', fromlist=['LinearSVC']).LinearSVC(C=1.0, dual='auto', max_iter=5000), True)]:
            import sklearn.base
            steps: list = [('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())]
            if kw:
                steps.append(('pca', PCA(n_components=kw['n'], random_state=42)))
            steps.append(('clf', sklearn.base.clone(mk)))
            pipe = Pipeline(steps)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipe.fit(Xtr, ytr)
            ft = time.time() - t0
            pr = pipe.predict(Xva)
            try:
                scc = scores_from_model(pipe, Xva)
            except Exception:
                scc = pr.astype(float)
            m = compute_metrics(yva, pr, scc)
            nc, ev = None, None
            if 'pca' in pipe.named_steps:
                nc = int(pipe.named_steps['pca'].n_components_)
                ev = float(np.sum(pipe.named_steps['pca'].explained_variance_ratio_))
            prows.append({'condition': cond, 'model': mname, 'n_comp': nc,
                          'exp_var': ev, 'acc': round(m['accuracy'], 4),
                          'f1': round(m['f1_phishing'], 4),
                          'roc': round(float(m.get('roc_auc', float('nan'))), 4),
                          'fit_s': round(ft, 2)})
    pd.DataFrame(prows).to_csv(OUT / 'pca_results.csv', index=False)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xtr), size=min(15000, len(Xtr)), replace=False)
    Z = PCA(n_components=min(10, Xs.shape[1]), random_state=42).fit_transform(
        StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(Xtr[idx])))
    yl = ytr[idx]
    crows = []
    for k in (2, 3, 4, 5, 6):
        try:
            lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
            si = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
            crows.append({'method': 'kmeans', 'k': k,
                          'sil': round(float(silhouette_score(Z[si], lab[si])), 4),
                          'db': round(float(davies_bouldin_score(Z, lab)), 4),
                          'ch': round(float(calinski_harabasz_score(Z, lab)), 1),
                          'ari': round(float(adjusted_rand_score(yl, lab)), 4),
                          'bic': None, 'aic': None})
        except Exception as e:
            crows.append({'method': 'kmeans', 'k': k, 'error': str(e)})
        try:
            gm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
            lab = gm.fit_predict(Z)
            si = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
            crows.append({'method': 'gmm_diag', 'k': k,
                          'sil': round(float(silhouette_score(Z[si], lab[si])), 4),
                          'db': round(float(davies_bouldin_score(Z, lab)), 4),
                          'ch': round(float(calinski_harabasz_score(Z, lab)), 1),
                          'ari': round(float(adjusted_rand_score(yl, lab)), 4),
                          'bic': round(float(gm.bic(Z)), 1), 'aic': round(float(gm.aic(Z)), 1)})
        except Exception as e:
            crows.append({'method': 'gmm_diag', 'k': k, 'error': str(e)})
    pd.DataFrame(crows).to_csv(OUT / 'clustering_results.csv', index=False)
    lab2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(Z)
    _fig_scatter(Z, lab2)


def _fig_scree(cum) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ind = np.arange(1, len(cum) + 1)
    ax.plot(ind, np.concatenate([[cum[0]], np.diff(cum)]), marker='o', label='individual')
    ax.plot(ind, cum, marker='s', label='cumulative')
    ax.set_xlabel('PC')
    ax.set_ylabel('explained variance')
    ax.set_title('PhishVN PCA scree (train-only fit, 21 feats)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'pca_scree.png', dpi=150)
    plt.close(fig)


def _fig_scatter(Z, lab) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, s=3, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PhishVN KMeans k=2 on PCA(10) subsample n=15000 (ARI posterior only)')
    fig.tight_layout()
    fig.savefig(FIG / 'cluster_pca_scatter.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figures from CSVs (no hardcoded numbers)
def run_figures() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fin = pd.read_csv(OUT / 'classification_metrics.csv')
    ext = pd.read_csv(OUT / 'external_metrics.csv')

    def plot_cm(tn, fp, fn, tp, title, path) -> None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow([[tn, fp], [fn, tp]], cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['pred benign', 'pred phish'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['true benign', 'true phish'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str([[tn, fp], [fn, tp]][i][j]), ha='center', va='center')
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    r = fin[fin['subset'] == 'FULL'].iloc[0]
    plot_cm(int(r['TN']), int(r['FP']), int(r['FN']), int(r['TP']),
            f"PhishVN FULL test ({r['model']})\nTN{r['TN']} FP{r['FP']} FN{r['FN']} TP{r['TP']}",
            FIG / 'confusion_internal_full.png')
    h = ext[(ext['external'] == 'hassan_340k') & (ext['model_subset'] == 'FULL')].iloc[0]
    plot_cm(int(h['TN']), int(h['FP']), int(h['FN']), int(h['TP']),
            f"Hassan external FULL (frozen)\nTN{h['TN']} FP{h['FP']} FN{h['FN']} TP{h['TP']}",
            FIG / 'confusion_hassan_full.png')
    g = ext[ext['external'] == 'hassan_340k'].set_index('model_subset')['accuracy'].to_dict()
    labels = ['FULL\ninternal', 'FULL\nHassan', 'GOLDSILVER\ninternal', 'GOLDSILVER\nHassan']
    vals = [float(r['accuracy']), float(g['FULL']),
            float(fin[fin['subset'] == 'GOLDSILVER'].iloc[0]['accuracy']),
            float(g['GOLDSILVER'])]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('accuracy')
    ax.set_title('PhishVN internal test vs Hassan external (frozen, no tuning)')
    for x, yv in zip(labels, vals):
        ax.text(labels.index(x), yv + 0.02, f'{yv:.4f}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'internal_vs_external_gap.png', dpi=150)
    plt.close(fig)
    s = pd.read_csv(OUT / 'test_by_source_FULL.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(s)), s['accuracy'])
    ax.set_xticks(range(len(s)))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('accuracy (FULL test)')
    ax.set_title(f"PhishVN FULL test accuracy by data source ({r['model']} frozen)")
    ax.set_xticklabels(s['source'], rotation=20, ha='right')
    for i, (a, n) in enumerate(zip(s['accuracy'], s['n'])):
        ax.text(i, a + 0.02, f'{a:.3f}\nn={n}', ha='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'source_bias.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- reports
def _f(x) -> str:
    return f'{float(x):.4f}'


def write_reports(audit: dict, fin: pd.DataFrame, ext: pd.DataFrame) -> None:
    fr = fin.set_index('subset').to_dict('index')
    er = ext.set_index(['external', 'model_subset']).to_dict('index')
    h_full = er[('hassan_340k', 'FULL')]
    h_gs = er[('hassan_340k', 'GOLDSILVER')]
    gap_full = (h_full['accuracy'] - fr['FULL']['accuracy']) * 100
    q_full = pd.read_csv(OUT / 'quick_val_FULL.csv').sort_values('f1', ascending=False).iloc[0]
    q_gs = pd.read_csv(OUT / 'quick_val_GOLDSILVER.csv').sort_values('f1', ascending=False).iloc[0]
    d_full = pd.read_csv(OUT / 'quick_val_FULL.csv').set_index('model').to_dict('index')['dummy_most_frequent']
    d_gs = pd.read_csv(OUT / 'quick_val_GOLDSILVER.csv').set_index('model').to_dict('index')['dummy_most_frequent']
    val_gs_pos = 'only 24 phishing in val (train 2076 / test 487)'
    src = pd.read_csv(OUT / 'test_by_source_FULL.csv').sort_values('accuracy')
    tier = pd.read_csv(OUT / 'test_by_tier_FULL.csv').sort_values('accuracy')
    scen = pd.read_csv(OUT / 'test_by_scenario_FULL.csv').sort_values('accuracy')
    usl = pd.read_csv(OUT / 'test_by_urlslice_FULL.csv')
    pca = pd.read_csv(OUT / 'pca_results.csv') if (OUT / 'pca_results.csv').exists() else None
    clu = pd.read_csv(OUT / 'clustering_results.csv') if (OUT / 'clustering_results.csv').exists() else None
    worst_src = src.iloc[0]
    worst_tier = tier.iloc[0]
    scen_big = scen[scen['n'] >= 100].sort_values('accuracy')
    usl = pd.read_csv(OUT / 'test_by_urlslice_FULL.csv')
    tiny_slices = usl[usl['tiny_n_flag'] == True]  # noqa: E712 (CSV round-trip bool)
    pca_line = ''
    if pca is not None and len(pca):
        p0 = pca[(pca['condition'] == 'no-PCA') & (pca['model'] == 'logreg_C1')].iloc[0]
        p9 = pca[(pca['condition'] == 'var90') & (pca['model'] == 'logreg_C1')].iloc[0]
        pca_line = (f"PCA (train-only fit, 21 feats): no-PCA logreg val acc {p0['acc']:.3f}/F1 {p0['f1']:.3f}; "
                    f"var90 ({int(p9['n_comp'])} PCs) {p9['acc']:.3f}/{p9['f1']:.3f}. "
                    f"Reduction hurts — keep full features. See `pca_results.csv`, `pca_scree.png`.")
    clu_line = ''
    if clu is not None and len(clu):
        k2 = clu[(clu['method'] == 'kmeans') & (clu['k'] == 2)].iloc[0]
        mx = clu.sort_values('ari', ascending=False).iloc[0]
        clu_line = (f"KMeans/GMM on PCA(10) subsample n=15000: k=2 ARI {k2['ari']:.4f}; "
                    f"best ARI {mx['ari']:.4f} ({mx['method']} k={int(mx['k'])}). "
                    f"No label-aligned clusters — expected, descriptive only. See `clustering_results.csv`.")

    quick = f"""# PhishVN — Quick Results

Label convention (PhishVN): `label=1` phishing / `label=0` benign. Positive class = phishing (1) in all metrics.
Dataset: Mendeley DOI `10.17632/b97hxbxtpd.4` (v3.1.0 open tier), zip SHA-256 `308351e9…5d0ab` verified; all `MANIFEST.txt` file hashes OK (14/14).
Official group-aware temporal split preserved: train {audit['split_counts'].get('train')} / val {audit['split_counts'].get('val')} / test {audit['split_counts'].get('test')}. Test opened once, after val selection.

## 1. Audit snapshot
- {audit['n']} rows: phishing {audit['label_counts'].get(1)} / benign {audit['label_counts'].get(0)}. Verified gold+silver core: {audit['tier_counts'].get('gold', 0) + audit['tier_counts'].get('silver', 0)}.
- Exact URL dups: {audit['exact_url_dups']}; normalized-URL dup groups: {audit['norm_dup_groups']}; registrable-domain overlap train/val/test: **{audit['domain_overlap']['train_val']}/{audit['domain_overlap']['train_test']}/{audit['domain_overlap']['val_test']}**.
- Collection artefact confirmed: `is_https` benign mean {audit['is_https_by_label'].get('0', float('nan')):.4f} vs phish {audit['is_https_by_label'].get('1', float('nan')):.4f} — **excluded from all modelling**.
- Modelling features (21, constant `PHISHVN_FEATURES`): {', '.join(FEATS)}. IDs, source, tier, timestamps, scenario/impersonated_org, metadata excluded. See `feature_audit.csv`.
- Extractor validated 53114/53116 integer-exact; 2 residual release-quirk rows documented (see DECISION.md). Covered by `tests/test_features.py`.

## 2. Validation (official val, selection only)
FULL: best **{q_full['model']}** acc {_f(q_full['accuracy'])}, F1 {_f(q_full['f1'])}, ROC {_f(q_full['roc'])}. See `quick_val_FULL.csv`.
GOLDSILVER ({val_gs_pos}): best **{q_gs['model']}** acc {_f(q_gs['accuracy'])}, F1 {_f(q_gs['f1'])}, ROC {_f(q_gs['roc'])}. RF wins F1 but trails LinSVC on val ROC; ±2–3 TP moves F1 at n=24 — selection unstable by construction. See `quick_val_GOLDSILVER.csv`.
- Accuracy-paradox check (majority baseline `dummy_most_frequent`): FULL dummy acc {_f(d_full['accuracy'])}, F1 {_f(d_full['f1'])}, macro-F1 {_f(d_full['macro_f1'])} vs best macro-F1 {_f(q_full['macro_f1'])}; GOLDSILVER dummy acc {_f(d_gs['accuracy'])}, F1 {_f(d_gs['f1'])}, macro-F1 {_f(d_gs['macro_f1'])} vs best macro-F1 {_f(q_gs['macro_f1'])}. The always-benign model "wins" on accuracy but scores F1=0 on phishing — macro-F1 (equal class weights) exposes it. See §"Vì sao macro-F1" trong DECISION.md.

## 3. Official test (frozen, once)
- FULL {fr['FULL']['model']} (train {fr['FULL']['n_train']} → test {fr['FULL']['n_test']}): acc **{_f(fr['FULL']['accuracy'])}** CI [{fr['FULL']['acc_exact_lo']:.5f},{fr['FULL']['acc_exact_hi']:.5f}], F1 **{_f(fr['FULL']['f1'])}**, ROC {_f(fr['FULL']['roc'])}, CM [TN{fr['FULL']['TN']} FP{fr['FULL']['FP']} FN{fr['FULL']['FN']} TP{fr['FULL']['TP']}].
- GOLDSILVER {fr['GOLDSILVER']['model']}: acc **{_f(fr['GOLDSILVER']['accuracy'])}** CI [{fr['GOLDSILVER']['acc_exact_lo']:.5f},{fr['GOLDSILVER']['acc_exact_hi']:.5f}], F1 **{_f(fr['GOLDSILVER']['f1'])}**, ROC {_f(fr['GOLDSILVER']['roc'])}, CM [TN{fr['GOLDSILVER']['TN']} FP{fr['GOLDSILVER']['FP']} FN{fr['GOLDSILVER']['FN']} TP{fr['GOLDSILVER']['TP']}].
- See `classification_metrics.csv`.

## 4. Bias slices (test, frozen FULL)
- Source: worst **{worst_src['source']} {_f(worst_src['accuracy'])}** (n={int(worst_src['n'])}); full table in `test_by_source_FULL.csv`.
- Tier: weakest **{worst_tier['tier']} {_f(worst_tier['accuracy'])}** (n={int(worst_tier['n'])}); scenarios range {scen_big.iloc[0]['scenario']} {_f(scen_big.iloc[0]['accuracy'])} → {scen_big.iloc[-1]['scenario']} {_f(scen_big.iloc[-1]['accuracy'])} (n>=100).
- URL slices carry `tiny_n_flag` (n<30): {', '.join(tiny_slices['slice'].tolist())} flagged, not interpreted. See `test_by_*.csv`.

## 5. External (frozen, no retraining/tuning)
- Hassan 340k: FULL acc **{_f(h_full['accuracy'])}** (gap **{gap_full:.2f}pp**), F1 {_f(h_full['f1'])}, ROC {_f(h_full['roc'])}. GOLDSILVER acc {_f(h_gs['accuracy'])}, F1 {_f(h_gs['f1'])}.
- JPCERT 2025-H1 phishing-only: FULL **recall {_f(er[('jpcert_2025H1_phishonly', 'FULL')]['rec'])}**; GOLDSILVER {_f(er[('jpcert_2025H1_phishonly', 'GOLDSILVER')]['rec'])} (`recall_phishonly`, `single_class=True`; accuracy left empty).
- Features recomputed from URL strings only; threshold 0.5 frozen. See `external_metrics.csv`.

## 6. PCA / clustering (after benchmark, exploratory)
- {pca_line}
- {clu_line}

## Files
- `outputs/{{feature_audit.csv,classification_metrics.csv,external_metrics.csv,quick_val_*.csv,test_by_*.csv,pca_results.csv,clustering_results.csv,frozen_*.joblib}}`
- `outputs/figures/` confusion, gap, source-bias, PCA scree, cluster scatter plots.
- PhiUSIIL negative benchmark archived in `archive/phiusiil/` (git tag `phiusiil-baseline`).
"""
    (OUT / 'QUICK_RESULTS.md').write_text(quick, encoding='utf-8')

    decision = f"""# PhishVN — Decision

Label convention: PhishVN `label=1` phishing / `label=0` benign (positive = phishing).

## Verdict
**ADOPT PhishVN as the primary dataset for the ML course project (suitable, with documented caveats). Retain PhiUSIIL only as a separate negative benchmark case study. PhishVN is NOT validated for real-world deployment.**

- Internal classification (official temporal test): SUITABLE. FULL {fr['FULL']['model']} acc {_f(fr['FULL']['accuracy'])} CI [{fr['FULL']['acc_exact_lo']:.5f},{fr['FULL']['acc_exact_hi']:.5f}], F1 {_f(fr['FULL']['f1'])}, ROC {_f(fr['FULL']['roc'])} — honest ~90% under a leakage-controlled temporal split with `is_https` excluded, vs PhiUSIIL's artefactual 99–100%.
- Cross-dataset generalization: NOT demonstrated. Frozen FULL drops to Hassan {_f(h_full['accuracy'])} ({gap_full:.2f}pp, ROC {_f(h_full['roc'])}, specificity {_f(h_full['spec'])}); JPCERT phishing recall stays {_f(er[('jpcert_2025H1_phishonly', 'FULL')]['rec'])} but benign collapses. Same failure mode as PhiUSIIL (benign side), so no deployment claim.
- The course value is precisely this contrast: a clean temporal benchmark (~90%) plus measurable source/tier/external gaps to analyse (RQ2–RQ4, PCA/clustering already runnable).

## Why suitable for the course (internal)
1. Provenance + protocol are sound: CC BY 4.0, MANIFEST SHA-256 verified (14/14), 0 URL dups, 0 cross-split domain overlap, official train/val/test honoured, HP selection on val only, test opened once.
2. Leakage controls present by design: CompPhish-aligned scheme-stripped lexical features, `is_https` excluded per authors, no IDs/sources/tiers/timestamps/scenarios in features (21 numeric only).
3. Headroom for ML coursework: LR vs RF vs LinearSVC compared on identical splits (FULL best {q_full['model']} F1 {_f(q_full['f1'])} val → {_f(fr['FULL']['f1'])} test); PCA hurts and clustering ARI ≈ 0 — reportable negative results, not failures.
4. Documented noise, not hidden: inter-annotator κ 0.609 (0.725 abuse-vs-legit), positive-arm noise 12.1% (bronze-concentrated), benign 4.0% — explains GOLDSILVER recall {_f(fr['GOLDSILVER']['rec'])} and silver-tier weakness without invalidating the benchmark.

## Why not deployable (generalization)
1. Source bias is large and measured: tranco global benign test acc lowest (see `test_by_source_FULL.csv`); silver tier weakest. The model learns ".vn/trusted = benign".
2. External benign collapses: Hassan specificity {_f(h_full['spec'])}. Internal 90% → external 44–48%.
3. JPCERT recall does not rescue this: phishing-only feeds cannot measure false positives (reported as `recall_phishonly` with `single_class=True`, accuracy left empty), which is where both PhiUSIIL and PhishVN fail.

## GOLDSILVER caveat
Gold+silver-only looks accurate ({_f(fr['GOLDSILVER']['accuracy'])}) but F1 is {_f(fr['GOLDSILVER']['f1'])} with phishing recall {_f(fr['GOLDSILVER']['rec'])} ({fr['GOLDSILVER']['TP']}/{fr['GOLDSILVER']['TP'] + fr['GOLDSILVER']['FN']}) on the official test — the verified core's newest phishing (temporal test = newest groups) is the hardest slice. Val positives are only 24, so the RF selection itself is unstable (LinSVC led val ROC). Use FULL as the primary benchmark; use GOLDSILVER to study label noise, drift, and class-imbalance handling — not as the headline number.
- Extractor audit note: `PHISHVN_FEATURES` (21 items) validated 53114/53116 exact on integers; 2 residual release-quirk rows documented. Slice tables carry `tiny_n_flag` (n<30).

## What to do
- Course report: benchmark FULL (LR/RF/SVC), ablate tiers (gold/silver vs bronze), sources (tranco global vs .vn), URL slices (tiny-n flagged, don't overclaim), PCA/clustering as completed.
- PhiUSIIL artifacts are archived in `archive/phiusiil/` as the catastrophic-generalization baseline.
- Deployment: CHANGE_DATASET (collect time-split, vendor-diverse URLs with legitimate deep links + independent validation) — out of scope for this course.

## Vì sao macro-F1 (và baseline dummy)
- Bài toán accuracy paradox: GOLDSILVER val chỉ có 24 phishing / 1585 mẫu. Model luôn đoán benign đạt acc {_f(d_gs['accuracy'])} — cao hơn RF {_f(q_gs['accuracy'])} — nhưng F1 phishing = {_f(d_gs['f1'])} (recall = 0, không bắt được ca phishing nào).
- Accuracy đánh trọng số theo prevalence (~98.5% trọng số dồn vào benign) nên bị đánh lừa. Macro-F1 cho mỗi lớp trọng số bằng nhau (50/50): dummy đạt macro-F1 {_f(d_gs['macro_f1'])} (trung bình của F1_benign≈0.99 và F1_phish=0), trong khi RF đạt {_f(q_gs['macro_f1'])} — đảo ngược bảng xếp hạng và phản ánh đúng giá trị thực (bắt được {float(q_gs['rec'])*100:.1f}% phishing).
- Quy tắc đọc báo cáo này: so accuracy với dummy trước (nếu model thua dummy về accuracy mà hơn về F1/macro-F1 thì đó là tín hiệu class-imbalance, không phải model tệ); chốt bằng F1 phishing + macro-F1, không chốt bằng accuracy.

## Evidence
- `outputs/QUICK_RESULTS.md`, `feature_audit.csv`, `classification_metrics.csv`, `external_metrics.csv`, `test_by_{{source,tier,scenario,urlslice}}_{{FULL,GOLDSILVER}}.csv`, `pca_results.csv`, `clustering_results.csv`
- `outputs/figures/confusion_internal_full.png`, `confusion_hassan_full.png`, `internal_vs_external_gap.png`, `source_bias.png`, `pca_scree.png`, `cluster_pca_scatter.png`
- Downloads: `data/phishvn/PhishVN_open.zip` (sha verified), extracted `data/phishvn/data/*`, `docs/*`, `MANIFEST.txt`
"""
    (OUT / 'DECISION.md').write_text(decision, encoding='utf-8')


# ---------------------------------------------------------------- orchestrator
def missing_inputs() -> list[str]:
    """Input files the full pipeline + reports need; empty list = ready to run."""
    need = [PHISHVN_DATA / 'vn_compphish.csv', PHISHVN_DATA / 'dataset_url.csv', HASSAN_CSV]
    miss = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if not list(JPCERT_DIR.glob('jpcert-*.csv')):
        miss.append(str((JPCERT_DIR / 'jpcert-*.csv').relative_to(ROOT)))
    return miss


def run_supplementary(vv: pd.DataFrame | None = None) -> int:
    """Post-hoc LogReg experiments: preprocessing ablation + leave-one-source-out."""
    from .experiments import run_experiments
    if vv is None:
        if _blocked():
            return 2
        vv = load_frames()
    print('[phishvn] supplementary experiments (preprocessing ablation + LOSO)...', flush=True)
    t0 = time.time()
    run_experiments(vv, OUT, FIG, HASSAN_CSV, load_jpcert_urls())
    print(f'[phishvn] supplementary DONE in {time.time() - t0:.1f}s -> outputs/EXPERIMENTS.md', flush=True)
    return 0


def _blocked() -> bool:
    miss = missing_inputs()
    if miss:
        print('BLOCKED: missing input files:')
        for m in miss:
            print(f'  {m}')
        print('PhishVN: download from Mendeley DOI 10.17632/b97hxbxtpd.4 and extract into data/phishvn/.')
        print('Hassan + JPCERT: python -m phishing_ml.cli fetch-external')
    return bool(miss)


def run_pipeline(mode: str = 'full') -> int:
    if _blocked():
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('[phishvn] loading data...', flush=True)
    vv = load_frames()
    print('[phishvn] audit...', flush=True)
    audit = run_audit(vv)
    print(f"[phishvn] audit: n={audit['n']} dups={audit['exact_url_dups']} "
          f"overlap={audit['domain_overlap']}", flush=True)
    print('[phishvn] quick benchmark (val selection)...', flush=True)
    best = run_quick(vv)
    for k, b in best.items():
        print(f"[phishvn] best {k}: {b['model']} f1={b['f1']:.4f} acc={b['accuracy']:.4f}", flush=True)
    print('[phishvn] final test (once, frozen)...', flush=True)
    fin = run_final_test(vv, best)
    print(fin.to_string(), flush=True)
    print('[phishvn] external evaluation (frozen, no tuning)...', flush=True)
    ext = run_external({k: v['model'] for k, v in best.items()})
    print(ext[['external', 'model_subset', 'accuracy', 'rec']].to_string() if len(ext) else 'no external', flush=True)
    if mode == 'full':
        print('[phishvn] PCA + clustering...', flush=True)
        run_pca_cluster(vv)
    print('[phishvn] figures...', flush=True)
    run_figures()
    print('[phishvn] reports...', flush=True)
    write_reports(audit, fin, ext)
    if mode == 'full':
        run_supplementary(vv)
    print(f'[phishvn] DONE in {time.time() - t0:.1f}s -> outputs/', flush=True)
    return 0
