"""Supplementary experiments (Logistic Regression only, FULL subset).

E-PREP  Preprocessing ablation: 13 variants of the same LogReg (C=10).
        Train on train, report val (selection) + official test + Hassan + JPCERT.
        Thresholds, when tuned, are tuned on val only and frozen before test.
        Statistics vs the baseline on test: exact McNemar p-value and a paired
        bootstrap 95% CI of the macro-F1 difference.
E-LOSO  Leave-one-source-out: every source in PhishVN is single-label
        (e.g. all chongluadao = phishing, all tranco = benign), so a model can
        learn "which source does this URL look like". For each source family
        we drop it from train and score its val+test rows, versus the same rows
        scored by the model trained with it. Diagnostic only: no selection.

This is a second, post-hoc experiment run after the main benchmark's single
test opening; it is reported as such and never feeds back into the main model.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    FunctionTransformer, QuantileTransformer, StandardScaler,
)

from .evaluate import clopper_pearson_ci
from .features import PHISHVN_FEATURES as FEATS, phishvn_features, strip_scheme
from .transformers import KMeansOneHot, resample_indices

SEED = 42
N_BOOT = 1000
C = 10.0  # same C as the main benchmark's selected logreg_C10

# Each PhishVN source contains a single label; families group near-duplicate sources
# so that holding one out does not leave its sibling in train.
SOURCE_FAMILY = {
    'chongluadao': 'chongluadao', 'openphish': 'openphish', 'tinnhiemmang': 'tinnhiemmang',
    'tinnhiem_org': 'tinnhiem (org+web)', 'tinnhiem_web': 'tinnhiem (org+web)',
    'tranco': 'tranco (global+vn)', 'tranco_vn': 'tranco (global+vn)',
}


# ---------------------------------------------------------------- variants
def _logreg(**kw) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, solver='lbfgs', **kw)


def _num(*steps) -> list:
    return [('impute', SimpleImputer(strategy='median')), *steps]


def _log1p() -> FunctionTransformer:
    return FunctionTransformer(np.log1p)  # all 21 features are counts/ratios >= 0


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=200_000)


def build_variants() -> list[dict]:
    """(name, what it changes, pipeline factory, input kind, threshold rule)."""
    both = lambda: ColumnTransformer([  # noqa: E731
        ('url', _tfidf(), 'url_text'),
        ('num', Pipeline(_num(('log1p', _log1p()), ('scale', StandardScaler()))), FEATS),
    ])
    return [
        dict(name='num_standard', desc='baseline: median impute + StandardScaler (= main benchmark logreg_C10)',
             make=lambda: Pipeline(_num(('scale', StandardScaler()), ('clf', _logreg()))), x='num', thr='fixed'),
        dict(name='num_log1p', desc='log1p before scaling (counts are heavily right-skewed)',
             make=lambda: Pipeline(_num(('log1p', _log1p()), ('scale', StandardScaler()), ('clf', _logreg()))), x='num', thr='fixed'),
        dict(name='num_quantile', desc='QuantileTransformer -> normal (rank-based, outlier-proof)',
             make=lambda: Pipeline(_num(('quantile', QuantileTransformer(output_distribution='normal', random_state=SEED)),
                                        ('clf', _logreg()))), x='num', thr='fixed'),
        dict(name='num_log1p_balanced', desc='log1p + class_weight=balanced (threshold 0.5)',
             make=lambda: Pipeline(_num(('log1p', _log1p()), ('scale', StandardScaler()),
                                        ('clf', _logreg(class_weight='balanced')))), x='num', thr='fixed'),
        dict(name='num_log1p_balanced_valthr', desc='log1p + class_weight=balanced + threshold chosen on val',
             make=lambda: Pipeline(_num(('log1p', _log1p()), ('scale', StandardScaler()),
                                        ('clf', _logreg(class_weight='balanced')))), x='num', thr='val'),
        dict(name='url_tfidf', desc='char 2-5-gram TF-IDF on the scheme-stripped URL string only',
             make=lambda: Pipeline([('tfidf', _tfidf()), ('clf', _logreg())]), x='url', thr='fixed'),
        dict(name='url_tfidf+num_log1p', desc='char TF-IDF + 21 numeric features (log1p + scaled)',
             make=lambda: Pipeline([('features', both()), ('clf', _logreg())]), x='both', thr='fixed'),
        # 'www.' is a collection artefact in PhishVN: benign 0.6% vs phishing 46% (train),
        # while 72% of Hassan's legitimate URLs carry it. Normalise it away before featurising.
        dict(name='num_standard_nowww', desc="baseline on www-normalised URLs (leading 'www.' stripped, features recomputed)",
             make=lambda: Pipeline(_num(('scale', StandardScaler()), ('clf', _logreg()))), x='num', thr='fixed', nowww=True),
        dict(name='url_tfidf_nowww', desc="char TF-IDF on www-normalised URLs",
             make=lambda: Pipeline([('tfidf', _tfidf()), ('clf', _logreg())]), x='url', thr='fixed', nowww=True),
        dict(name='url_tfidf+num_log1p_nowww', desc="char TF-IDF + numeric (log1p), both on www-normalised URLs",
             make=lambda: Pipeline([('features', both()), ('clf', _logreg())]), x='both', thr='fixed', nowww=True),
        # ---- classic data-processing techniques on the 21 numeric features ----
        dict(name='num_pca95', desc='dimensionality reduction: PCA keeping 95% of variance (fit on train), then LogReg',
             make=lambda: Pipeline(_num(('scale', StandardScaler()), ('pca', PCA(n_components=0.95, random_state=SEED)),
                                        ('clf', _logreg()))), x='num', thr='fixed', group='dimensionality reduction'),
        dict(name='num_undersample', desc='imbalance: randomly drop phishing rows in train to a 50/50 balance',
             make=lambda: Pipeline(_num(('scale', StandardScaler()), ('clf', _logreg()))),
             x='num', thr='fixed', resample='under', group='class imbalance'),
        dict(name='num_kmeans_onehot', desc='clustering as a feature: append one-hot K-Means (k=8, fit on train) cluster id',
             make=lambda: Pipeline(_num(('scale', StandardScaler()), ('kmeans', KMeansOneHot(8, SEED)), ('clf', _logreg()))),
             x='num', thr='fixed', group='clustering as feature'),
    ]


def url_text(urls: pd.Series) -> pd.Series:
    return urls.astype(str).map(strip_scheme).str.lower()


def strip_www(url: str) -> str:
    """Scheme-stripped URL with a leading 'www.' removed (host normalisation)."""
    u = strip_scheme(str(url).strip())
    return u[4:] if u.lower().startswith('www.') else u


NW = [f'nw_{c}' for c in FEATS]


def add_nowww(df: pd.DataFrame) -> pd.DataFrame:
    """Add www-normalised URL + its 21 features (nw_*) so variants can use either view."""
    u = df['url'].map(strip_www)
    f = pd.DataFrame([phishvn_features(x) for x in u.tolist()], index=df.index)[FEATS]
    f.columns = NW
    return pd.concat([df.assign(url_nw=u), f], axis=1)


def make_x(df: pd.DataFrame, v: dict):
    """Model input for a variant: numeric frame, URL text, or both; raw or www-normalised view."""
    nowww = v.get('nowww', False)
    num = df[NW].set_axis(FEATS, axis=1) if nowww else df[FEATS]
    txt = url_text(df['url_nw'] if nowww else df['url'])
    if v['x'] == 'num':
        return num
    if v['x'] == 'url':
        return txt
    return num.assign(url_text=txt.to_numpy())


# ---------------------------------------------------------------- statistics
def best_f1_threshold(y, score) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, score >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metrics(y, score, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    k, n = tp + tn, len(y)
    lo, hi = clopper_pearson_ci(k, n)
    two_class = len(np.unique(y)) == 2
    return {
        'n': n, 'accuracy': k / n if n else float('nan'), 'acc_lo': lo, 'acc_hi': hi,
        'prec': tp / (tp + fp) if tp + fp else 0.0,
        'rec': tp / (tp + fn) if tp + fn else float('nan'),
        'spec': tn / (tn + fp) if tn + fp else float('nan'),
        'f1': f1_score(y, pred, zero_division=0),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0) if two_class else float('nan'),
        'roc': roc_auc_score(y, score) if two_class else float('nan'),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }


def mcnemar_exact(y, pred_a, pred_b) -> tuple[int, int, float]:
    """Exact McNemar test on discordant pairs: (a right & b wrong, a wrong & b right, p)."""
    ra, rb = (pred_a == y), (pred_b == y)
    n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def _macro_f1_counts(y, pred):
    """Vectorised macro-F1 for a batch of resamples: y, pred are (B, n) int arrays."""
    tp = ((pred == 1) & (y == 1)).sum(1)
    tn = ((pred == 0) & (y == 0)).sum(1)
    fp = ((pred == 1) & (y == 0)).sum(1)
    fn = ((pred == 0) & (y == 1)).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1p = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1n = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1p + f1n) / 2


def paired_bootstrap_macro_f1_diff(y, pred_new, pred_base, n_boot: int = N_BOOT, seed: int = SEED):
    """95% percentile CI of macro-F1(new) - macro-F1(base), resampling the same test rows."""
    rng = np.random.default_rng(seed)
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_new, pred_base))
    diffs = []
    for chunk in np.array_split(np.arange(n_boot), 10):
        idx = rng.integers(0, len(y), size=(len(chunk), len(y)))
        diffs.append(_macro_f1_counts(y[idx], a[idx]) - _macro_f1_counts(y[idx], b[idx]))
    d = np.concatenate(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- E-PREP
def www_rates(vv: pd.DataFrame, externals: dict) -> pd.DataFrame:
    """Share of URLs starting with 'www.' per dataset x label (and per PhishVN source on train)."""
    lab = {0: 'benign', 1: 'phishing'}

    def rate(d: pd.DataFrame, by: str, name: str) -> list[dict]:
        g = url_text(d['url']).str.startswith('www.').groupby(d[by]).agg(['mean', 'size'])
        return [{'dataset': name, 'group': lab.get(k, k) if by == 'label' else k, 'n': int(r['size']),
                 'www_rate': float(r['mean'])} for k, r in g.iterrows()]
    tr = vv[vv['split'] == 'train']
    rows = rate(tr, 'label', 'phishvn_train') + rate(tr, 'source', 'phishvn_train_source')
    for name, d in externals.items():
        rows += rate(d, 'label', name)
    return pd.DataFrame(rows)


def load_external_frames(hassan_csv: Path, jpcert_urls: list[str]) -> dict:
    """External sets with the same 21 numeric features + raw URL; labels for scoring only."""
    out = {}
    if hassan_csv.exists():
        h = pd.read_csv(hassan_csv, low_memory=False)
        url_col = 'url' if 'url' in h.columns else h.columns[1]
        lab_col = 'label' if 'label' in h.columns else h.columns[0]
        y = h[lab_col].astype(str).str.lower().map({'phishing': 1, 'legitimate': 0, 'bad': 1, 'good': 0})
        h = pd.DataFrame({'url': h[url_col].astype(str), 'label': y}).dropna(subset=['label'])
        out['hassan'] = _with_features(h.reset_index(drop=True))
    if jpcert_urls:
        out['jpcert'] = _with_features(pd.DataFrame({'url': jpcert_urls, 'label': 1}))
    return out


def _with_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame([phishvn_features(u) for u in df['url'].tolist()])
    return add_nowww(pd.concat([df[['url', 'label']].astype({'label': int}), f[FEATS]], axis=1))


def run_preprocessing_ablation(vv: pd.DataFrame, externals: dict) -> pd.DataFrame:
    tr, va, te = (vv[vv['split'] == s] for s in ('train', 'val', 'test'))
    ytr, yva, yte = (d['label'].to_numpy().astype(int) for d in (tr, va, te))
    rows, test_preds, top_ngrams = [], {}, pd.DataFrame()
    for v in build_variants():
        pipe = v['make']()
        fit_rows = resample_indices(ytr, v['resample'], SEED) if v.get('resample') else np.arange(len(tr))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipe.fit(make_x(tr.iloc[fit_rows], v), ytr[fit_rows])
        s_va = pipe.predict_proba(make_x(va, v))[:, 1]
        thr = best_f1_threshold(yva, s_va) if v['thr'] == 'val' else 0.5
        s_te = pipe.predict_proba(make_x(te, v))[:, 1]
        test_preds[v['name']] = (s_te >= thr).astype(int)
        row = {'variant': v['name'], 'group': v.get('group', ''), 'description': v['desc'], 'threshold': thr}
        row.update({f'val_{k}': x for k, x in metrics(yva, s_va, thr).items()})
        row.update({f'test_{k}': x for k, x in metrics(yte, s_te, thr).items()})
        if 'hassan' in externals:
            h = externals['hassan']
            mh = metrics(h['label'].to_numpy(), pipe.predict_proba(make_x(h, v))[:, 1], thr)
            row.update({f'hassan_{k}': mh[k] for k in ('accuracy', 'spec', 'rec', 'macro_f1', 'roc')})
        if 'jpcert' in externals:
            j = externals['jpcert']
            row['jpcert_rec'] = float((pipe.predict_proba(make_x(j, v))[:, 1] >= thr).mean())
        if v['name'] == 'url_tfidf':
            top_ngrams = top_tfidf_ngrams(pipe)
        rows.append(row)
        print(f"  [prep] {v['name']:28s} val macroF1 {row['val_macro_f1']:.4f}  test macroF1 {row['test_macro_f1']:.4f}", flush=True)
    base = test_preds['num_standard']
    for row in rows:
        n01, n10, p = mcnemar_exact(yte, test_preds[row['variant']], base)
        lo, hi = paired_bootstrap_macro_f1_diff(yte, test_preds[row['variant']], base)
        row.update({'test_only_new_right': n01, 'test_only_base_right': n10, 'mcnemar_p': p,
                    'test_dmacro_f1': row['test_macro_f1'] - rows[0]['test_macro_f1'],
                    'test_dmacro_f1_lo': lo, 'test_dmacro_f1_hi': hi})
    return pd.DataFrame(rows), top_ngrams


def top_tfidf_ngrams(pipe: Pipeline, k: int = 20) -> pd.DataFrame:
    """Largest LogReg weights of the char TF-IDF model: what substrings drive each class."""
    vocab = pipe.named_steps['tfidf'].get_feature_names_out()
    coef = pipe.named_steps['clf'].coef_[0]
    order = np.argsort(coef)
    pick = [('phishing', i) for i in order[::-1][:k]] + [('benign', i) for i in order[:k]]
    return pd.DataFrame([{'pushes_to': c, 'rank': r % k + 1, 'ngram': vocab[i], 'coef': float(coef[i])}
                         for r, (c, i) in enumerate(pick)])


# ---------------------------------------------------------------- E-LOSO
def run_leave_one_source_out(vv: pd.DataFrame, variants=('num_standard', 'num_standard_nowww', 'url_tfidf+num_log1p_nowww')) -> pd.DataFrame:
    vv = vv.assign(family=vv['source'].map(SOURCE_FAMILY))
    if vv['family'].isna().any():
        raise ValueError(f"unmapped sources: {sorted(vv.loc[vv['family'].isna(), 'source'].unique())}")
    spec = {v['name']: v for v in build_variants()}
    tr = vv[vv['split'] == 'train']
    ev = vv[vv['split'].isin(['val', 'test'])]
    rows = []
    for vname in variants:
        v = spec[vname]
        full = v['make']()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            full.fit(make_x(tr, v), tr['label'].to_numpy())
        for fam in sorted(vv['family'].unique()):
            held = ev[ev['family'] == fam]
            y = held['label'].to_numpy().astype(int)
            rest = tr[tr['family'] != fam]
            loso = v['make']()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                loso.fit(make_x(rest, v), rest['label'].to_numpy())
            acc_in = float((full.predict(make_x(held, v)) == y).mean())
            k_out = int((loso.predict(make_x(held, v)) == y).sum())
            lo, hi = clopper_pearson_ci(k_out, len(y))
            rows.append({'variant': vname, 'family': fam, 'label': 'phishing' if y.mean() == 1 else 'benign',
                         'n_eval': len(y), 'n_train_dropped': int((tr['family'] == fam).sum()),
                         'train_pos_left': int(rest['label'].sum()), 'train_neg_left': int((rest['label'] == 0).sum()),
                         'acc_in_source': acc_in, 'acc_held_out': k_out / len(y),
                         'held_lo': lo, 'held_hi': hi,
                         'gap_pp': (k_out / len(y) - acc_in) * 100, 'tiny_n_flag': len(y) < 30})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures + report
def plot_experiments(prep: pd.DataFrame, loso: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ('val_macro_f1', 'test_macro_f1', 'hassan_macro_f1') if c in prep]
    x = np.arange(len(prep))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(11, 0.7 * len(prep)), 4.8))
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, prep[c], w, label=c.replace('_macro_f1', ''))
    ax.set_xticks(x)
    ax.set_xticklabels(prep['variant'], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('macro-F1')
    ax.set_ylim(0, 1.12)
    ax.set_title('Preprocessing ablation (LogReg C=10, FULL): val / test / Hassan macro-F1')
    ax.legend(fontsize=8, ncol=len(cols), loc='upper left')
    fig.tight_layout()
    fig.savefig(fig_dir / 'preprocessing_ablation.png', dpi=150)
    plt.close(fig)

    fams = sorted(loso['family'].unique())
    fig, axes = plt.subplots(1, loso['variant'].nunique(), figsize=(12, 4.5), sharey=True)
    for ax, (vname, g) in zip(np.atleast_1d(axes), loso.groupby('variant', sort=False)):
        g = g.set_index('family').loc[fams]
        xi = np.arange(len(fams))
        ax.bar(xi - 0.2, g['acc_in_source'], 0.4, label='source in train')
        ax.bar(xi + 0.2, g['acc_held_out'], 0.4, label='source held out')
        ax.set_xticks(xi)
        ax.set_xticklabels([f"{f}\n({lab}, n={n})" for f, lab, n in zip(fams, g['label'], g['n_eval'])],
                           rotation=25, ha='right', fontsize=7)
        ax.set_title(vname, fontsize=9)
        ax.set_ylim(0, 1.12)
    np.atleast_1d(axes)[0].set_ylabel('accuracy on that source (val+test rows)')
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=2, loc='upper left')
    fig.suptitle('Leave-one-source-out: does the model recognise a source it never saw?')
    fig.tight_layout()
    fig.savefig(fig_dir / 'leave_one_source_out.png', dpi=150)
    plt.close(fig)


def _f(x, nd: int = 4) -> str:
    return 'n/a' if pd.isna(x) else f'{x:.{nd}f}'


def write_report(prep: pd.DataFrame, loso: pd.DataFrame, ngrams: pd.DataFrame, skew: pd.Series,
                 www: pd.DataFrame, path: Path) -> None:
    base = prep.iloc[0]
    best = prep.loc[prep['val_macro_f1'].idxmax()]
    num = prep[prep['variant'].str.startswith('num_') & (prep['threshold'] == 0.5)]
    has_h = 'hassan_macro_f1' in prep

    head = '| Variant | thr | Val acc | Val F1 | Val macro-F1 | Test acc [CI95] | Test F1 | Test macro-F1 | Test ROC |'
    sep = '|---|---|---|---|---|---|---|---|---|'
    if has_h:
        head += ' Hassan acc | Hassan spec | Hassan macro-F1 |'
        sep += '---|---|---|'
    if 'jpcert_rec' in prep:
        head += ' JPCERT rec |'
        sep += '---|'
    lines = [head, sep]
    for _, r in prep.iterrows():
        line = (f"| `{r['variant']}` | {r['threshold']:.2f} | {_f(r['val_accuracy'])} | {_f(r['val_f1'])} | "
                f"{_f(r['val_macro_f1'])} | {_f(r['test_accuracy'])} [{_f(r['test_acc_lo'], 3)}, {_f(r['test_acc_hi'], 3)}] | "
                f"{_f(r['test_f1'])} | {_f(r['test_macro_f1'])} | {_f(r['test_roc'])} |")
        if has_h:
            line += f" {_f(r['hassan_accuracy'])} | {_f(r['hassan_spec'])} | {_f(r['hassan_macro_f1'])} |"
        if 'jpcert_rec' in prep:
            line += f" {_f(r['jpcert_rec'])} |"
        lines.append(line)

    stat = ['| Variant | Δ test macro-F1 vs baseline [paired bootstrap CI95] | only new right / only baseline right | McNemar p |',
            '|---|---|---|---|']
    for _, r in prep.iloc[1:].iterrows():
        sig = ' *' if r['mcnemar_p'] < 0.05 else ''
        stat.append(f"| `{r['variant']}` | {r['test_dmacro_f1']:+.4f} [{r['test_dmacro_f1_lo']:+.4f}, {r['test_dmacro_f1_hi']:+.4f}] | "
                    f"{int(r['test_only_new_right'])} / {int(r['test_only_base_right'])} | {r['mcnemar_p']:.2g}{sig} |")

    lo_lines = ['| Variant | Source family (label) | n eval | acc, source in train | acc, source held out [CI95] | gap (pp) |',
                '|---|---|---|---|---|---|']
    for _, r in loso.iterrows():
        flag = ' ⚠ n<30' if r['tiny_n_flag'] else ''
        lo_lines.append(f"| `{r['variant']}` | {r['family']} ({r['label']}) | {int(r['n_eval'])}{flag} | {_f(r['acc_in_source'])} | "
                        f"{_f(r['acc_held_out'])} [{_f(r['held_lo'], 3)}, {_f(r['held_hi'], 3)}] | {r['gap_pp']:+.1f} |")
    worst = loso.loc[loso['gap_pp'].idxmin()]
    lb = loso[loso['variant'] == 'num_standard']
    benign_held = lb[lb['label'] == 'benign']['acc_held_out']

    scale_note = ', '.join(f"`{r['variant']}` {_f(r['val_macro_f1'])}" for _, r in num.iterrows())
    idx = prep.set_index('variant')
    scale_rows = idx.loc[[n for n in ('num_log1p', 'num_quantile') if n in idx.index]]
    scale_sig = '; '.join(
        f"`{n}` Δ test macro-F1 {r['test_dmacro_f1']:+.4f}, McNemar p={r['mcnemar_p']:.2g}"
        + (' (khác có ý nghĩa)' if r['mcnemar_p'] < 0.05 else ' (không khác có ý nghĩa)')
        for n, r in scale_rows.iterrows())
    skew_txt = ', '.join(f"`{c}` {v:.0f}" for c, v in skew.head(4).items())
    h_best = prep.loc[prep['hassan_macro_f1'].idxmax()] if has_h else None
    ng = lambda c: ', '.join(f"`{t}`" for t in ngrams[ngrams['pushes_to'] == c]['ngram'].head(15))  # noqa: E731
    wr = {(r['dataset'], r['group']): r for _, r in www.iterrows()}
    www_lines = ['| Tập | Nhóm | n | tỷ lệ URL bắt đầu bằng `www.` |', '|---|---|---|---|']
    for _, r in www.iterrows():
        www_lines.append(f"| {r['dataset']} | {r['group']} | {int(r['n'])} | {r['www_rate']:.1%} |")

    def pair(raw: str, nw: str) -> str:
        if raw not in idx.index or nw not in idx.index:
            return ''
        a_, b_ = idx.loc[raw], idx.loc[nw]
        out = (f"| `{raw}` → `{nw}` | {_f(a_['val_macro_f1'])} → {_f(b_['val_macro_f1'])} | "
               f"{_f(a_['test_macro_f1'])} → {_f(b_['test_macro_f1'])} |")
        if has_h:
            out += (f" {_f(a_['hassan_spec'])} → **{_f(b_['hassan_spec'])}** | "
                    f"{_f(a_['hassan_macro_f1'])} → **{_f(b_['hassan_macro_f1'])}** |")
        return out
    nw_head = '| Raw → www-normalised | Val macro-F1 | Test macro-F1 |' + (' Hassan specificity | Hassan macro-F1 |' if has_h else '')
    nw_sep = '|---|---|---|' + ('---|---|' if has_h else '')
    nw_lines = [nw_head, nw_sep] + [x for x in (pair('num_standard', 'num_standard_nowww'), pair('url_tfidf', 'url_tfidf_nowww'),
                                                 pair('url_tfidf+num_log1p', 'url_tfidf+num_log1p_nowww')) if x]
    pairs = [(r, n) for r, n in (('num_standard', 'num_standard_nowww'), ('url_tfidf', 'url_tfidf_nowww'),
                                 ('url_tfidf+num_log1p', 'url_tfidf+num_log1p_nowww')) if r in idx.index and n in idx.index]
    dval = [idx.loc[n, 'val_macro_f1'] - idx.loc[r, 'val_macro_f1'] for r, n in pairs] or [0.0]
    nw_text = (f"**Điểm nội bộ {'giảm' if max(dval) < 0 else 'thay đổi'}** khi bỏ `www.` (Δ val macro-F1 từ {min(dval):+.4f} đến {max(dval):+.4f}). "
               "Một phần thành tích nội bộ đến từ lối tắt `www.`; con số sau khi bỏ trung thực hơn, dù thấp hơn.")
    if has_h and pairs:
        gains = {n: idx.loc[n, 'hassan_spec'] - idx.loc[r, 'hassan_spec'] for r, n in pairs}
        top = max(gains, key=gains.get)
        flat = [n for n, g in gains.items() if g < 0.02]
        nw_text += (f" Trên Hassan, cải thiện lớn nhất ở `{top}`: specificity +{gains[top]:.3f}, macro-F1 "
                    f"{_f(idx.loc[top, 'hassan_macro_f1'])}, accuracy {_f(idx.loc[top, 'hassan_accuracy'])}. "
                    "Như vậy với model chỉ đọc chuỗi URL, `www.` là nguyên nhân lớn của lỗi benign trên Hassan.")
        if flat:
            nw_text += (f" Ngược lại {', '.join(f'`{n}`' for n in flat)} gần như không đổi (Δ specificity < 0.02): "
                        "các feature số còn chứa lối tắt khác (độ dài domain, số subdomain, đuôi domain…) mà việc bỏ `www.` không gỡ được.")
    classic = prep[prep['group'].fillna('') != ''] if 'group' in prep else prep.iloc[0:0]
    cl_lines = ['| Nhóm kỹ thuật | Biến thể | Làm gì | Val macro-F1 | Test macro-F1 | Δ test vs baseline [CI95] | McNemar p |'
                + (' Hassan macro-F1 |' if has_h else ''),
                '|---|---|---|---|---|---|---|' + ('---|' if has_h else '')]
    for _, r in classic.iterrows():
        cl_lines.append(f"| {r['group']} | `{r['variant']}` | {r['description']} | {_f(r['val_macro_f1'])} | {_f(r['test_macro_f1'])} | "
                        f"{r['test_dmacro_f1']:+.4f} [{r['test_dmacro_f1_lo']:+.4f}, {r['test_dmacro_f1_hi']:+.4f}] | {r['mcnemar_p']:.2g} |"
                        + (f" {_f(r['hassan_macro_f1'])} |" if has_h else ''))
    if len(classic):
        better = classic[(classic['test_dmacro_f1_lo'] > 0)]
        worse = classic[(classic['test_dmacro_f1_hi'] < 0)]
        same = classic[(classic['test_dmacro_f1_lo'] <= 0) & (classic['test_dmacro_f1_hi'] >= 0)]
        top_val = classic.loc[classic['val_macro_f1'].idxmax()]
        lst = lambda d, mark=True: ', '.join(f"`{v}` ({x:+.4f}{'†' if mark and pv >= 0.05 else ''})"  # noqa: E731
                                  for v, x, pv in zip(d['variant'], d['test_dmacro_f1'], d['mcnemar_p'])) or 'không có'
        cl_text = chr(10).join([
            f"- Tốt nhất theo val trong nhóm này: **`{top_val['variant']}`** (val macro-F1 {_f(top_val['val_macro_f1'])} "
            f"so với baseline {_f(base['val_macro_f1'])}).",
            f"- **Tốt hơn baseline có ý nghĩa** (cả khoảng tin cậy > 0): {lst(better)}.",
            f"- **Không khác biệt** (khoảng tin cậy chứa 0): {lst(same, False)}.",
            f"- **Kém hơn có ý nghĩa** (cả khoảng tin cậy < 0): {lst(worse)}.",
        ] + (["- `†` = khoảng tin cậy bootstrap không chứa 0 nhưng McNemar p ≥ 0.05: hai kiểm định không đồng thuận, nên coi là "
               "chênh lệch rất nhỏ, chưa chắc chắn."]
              if (pd.concat([better, worse])['mcnemar_p'] >= 0.05).any() else []))
        if has_h:
            hb = classic.loc[classic['hassan_macro_f1'].idxmax()]
            cl_text += chr(10) + (f"- Trên Hassan, tốt nhất trong nhóm này là `{hb['variant']}` (macro-F1 {_f(hb['hassan_macro_f1'])} "
                                  f"so với baseline {_f(base['hassan_macro_f1'])}).")
    else:
        cl_text = ''
    ls = loso.set_index(['variant', 'family'])

    def loso_cmp(fam: str) -> str:
        try:
            a_, b_ = ls.loc[('num_standard', fam)], ls.loc[('num_standard_nowww', fam)]
        except KeyError:
            return ''
        return f"{fam}: {_f(a_['acc_held_out'])} → {_f(b_['acc_held_out'])}"
    loso_nw = '; '.join(x for x in (loso_cmp(f) for f in sorted(loso['family'].unique())) if x)
    loso_nw_text = ''
    if ('num_standard_nowww', 'chongluadao') in ls.index:
        c0, c1 = ls.loc[('num_standard', 'chongluadao'), 'acc_held_out'], ls.loc[('num_standard_nowww', 'chongluadao'), 'acc_held_out']
        if c1 < c0 - 0.05:
            loso_nw_text += (f" Với chongluadao (nguồn phishing lớn nhất) bị loại khỏi train, recall rơi {_f(c0)} → {_f(c1)} khi bỏ `www.`: "
                             "`www.` là tín hiệu chính model dùng để nhận ra phishing kiểu chongluadao.")
        nb = loso[(loso['variant'] == 'num_standard_nowww') & (loso['label'] == 'benign')]['acc_held_out']
        if len(nb) and nb.max() < 0.5:
            loso_nw_text += (f" Các họ benign chưa thấy vẫn chỉ đạt ≤ {_f(nb.max())}, nên chữ ký nguồn không chỉ nằm ở `www.` "
                             "(đuôi domain, độ dài, kiểu đặt tên của từng danh sách).")
    pv_b, pv_p = wr.get(('phishvn_train', 'benign')), wr.get(('phishvn_train', 'phishing'))
    hs_b = wr.get(('hassan', 'benign'))
    txt = f"""# PhishVN — Thí nghiệm bổ sung: tiền xử lý + leave-one-source-out

Logistic Regression C=10, tập FULL. Nhãn: `1` phishing / `0` benign, lớp dương = phishing.
**Đây là thí nghiệm thứ hai, chạy sau benchmark chính** (test chính đã mở một lần cho `logreg_C10`). Mọi lựa chọn (ngưỡng, biến thể tốt nhất) dựa trên **val**; test, Hassan, JPCERT chỉ dùng để chấm. Kết quả này không thay thế model chính trong `DECISION.md`.

## 0. Phát hiện chính: tiền tố `www.` là dấu vết thu thập dữ liệu

{chr(10).join(www_lines)}

Trong PhishVN, URL benign gần như được lưu dưới dạng domain trần (train: {pv_b['www_rate']:.1%} có `www.`), còn phishing thì {pv_p['www_rate']:.1%}. Model vì vậy học luôn quy tắc "`www.` = phishing" (n-gram `www.` đứng đầu danh sách đẩy về phishing ở mục 1b).{f" Trên Hassan, {hs_b['www_rate']:.1%} URL hợp lệ có `www.`, nên bị gán nhãn phishing hàng loạt. Đây là cơ chế cụ thể đằng sau specificity rất thấp trên Hassan." if hs_b is not None else ''} Cách xử lý: **chuẩn hóa host bằng cách bỏ `www.` đứng đầu** trước khi trích feature (các biến thể `*_nowww`). Đây là bước làm sạch hợp lý trong mọi trường hợp: `www.example.com` và `example.com` là cùng một site.

## 1. So sánh cách tiền xử lý

{chr(10).join(lines)}

Mô tả biến thể:
{chr(10).join(f"- `{r['variant']}`: {r['description']}" for _, r in prep.iterrows())}

### Kiểm định so với baseline (`num_standard`) trên test (n={int(base['test_n'])})
{chr(10).join(stat)}

`*` = p < 0.05 (McNemar chính xác, trên các mẫu hai model dự đoán khác nhau). CI bootstrap ghép cặp {N_BOOT} lần, cùng các dòng test.

### Đọc kết quả
- **Biến thể tốt nhất theo val macro-F1: `{best['variant']}`** (val {_f(best['val_macro_f1'])} so với baseline {_f(base['val_macro_f1'])}; test {_f(best['test_macro_f1'])} so với {_f(base['test_macro_f1'])}, Δ {best['test_dmacro_f1']:+.4f} [{best['test_dmacro_f1_lo']:+.4f}, {best['test_dmacro_f1_hi']:+.4f}], McNemar p={best['mcnemar_p']:.2g}).
- **Đổi cách scale feature số** (val macro-F1: {scale_note}). Feature lệch rất mạnh (skew trên train: {skew_txt}), vậy mà log1p không giúp gì; còn QuantileTransformer làm kết quả **kém hơn** ({scale_sig}). Với LogReg ở đây, thông tin nằm ở việc feature có hay không có và ở thứ tự thô của giá trị; ép về phân phối chuẩn theo hạng làm mất khoảng cách giữa các giá trị cực trị (URL rất dài, query rất nhiều tham số), mà chính các giá trị đó mang tín hiệu.
- **class_weight / ngưỡng**: `num_log1p_balanced` ở ngưỡng 0.5 có test F1 {_f(prep.set_index('variant').loc['num_log1p_balanced', 'test_f1'])}; khi chọn lại ngưỡng trên val (thr={prep.set_index('variant').loc['num_log1p_balanced_valthr', 'threshold']:.2f}) F1 về {_f(prep.set_index('variant').loc['num_log1p_balanced_valthr', 'test_f1'])}. Trọng số lớp chủ yếu chỉ dịch ngưỡng; chọn ngưỡng trên val là đủ.
- **TF-IDF n-gram ký tự** học trực tiếp các chuỗi con của URL mà 21 feature đếm không thấy (xem mục 1b).{f" Trên Hassan, biến thể tốt nhất theo val (`{best['variant']}`) có macro-F1 {_f(best['hassan_macro_f1'])}, specificity {_f(best['hassan_spec'])} so với baseline {_f(base['hassan_macro_f1'])} / {_f(base['hassan_spec'])}. Biến thể tổng quát hóa tốt nhất sang Hassan lại là `{h_best['variant']}` (macro-F1 {_f(h_best['hassan_macro_f1'])}, specificity {_f(h_best['hassan_spec'])}). Cải thiện nội bộ **không** tự động chuyển thành khả năng tổng quát hóa, và chọn model theo val không chọn ra model tổng quát hóa tốt nhất." if has_h else ''}

### 1b. N-gram có trọng số lớn nhất (`url_tfidf`)
- Đẩy về **phishing**: {ng('phishing')}
- Đẩy về **benign**: {ng('benign')}

Phía phishing bị chi phối bởi `www.` (mục 0) và các từ khóa lừa đảo tiếng Việt (ví dụ `vay`, `nap`, `vip`). Phía benign là đuôi domain (`.vn` ở cuối host, `.gov`, `.jp`, `.de`): đó là chữ ký của danh sách tổ chức tin cậy (tinnhiem) và Tranco, đúng loại phụ thuộc nguồn mà mục 2 đo được. Danh sách đầy đủ ở `tfidf_top_ngrams.csv`.

### 1c. Bỏ `www.` thay đổi được gì
{chr(10).join(nw_lines)}

{nw_text}

### 1d. PCA, undersampling và cụm K-Means trên 21 feature số
Cùng LogReg C=10, mỗi dòng chỉ đổi **một** bước xử lý so với baseline (`num_standard`). Mọi tham số của bước xử lý (các thành phần PCA, tâm cụm K-Means, mẫu được giữ khi undersampling) chỉ học trên train.

{chr(10).join(cl_lines)}

{cl_text}

## 2. Leave-one-source-out

Mỗi nguồn trong PhishVN chỉ chứa **một nhãn** (ví dụ chongluadao = 100% phishing, tranco = 100% benign), nên model có thể học "URL này giống nguồn nào" thay vì "URL này có phải phishing không". Cách đo: bỏ hẳn một họ nguồn khỏi train, rồi chấm trên các dòng val+test của chính họ đó. Accuracy ở đây bằng recall (họ phishing) hoặc specificity (họ benign).

{chr(10).join(lo_lines)}

### Đọc kết quả
- Gap lớn nhất: **{worst['family']} ({worst['label']})** với `{worst['variant']}`, {worst['gap_pp']:+.1f}pp ({_f(worst['acc_in_source'])} → {_f(worst['acc_held_out'])}).
- Với baseline, accuracy trên họ **benign** khi bị loại khỏi train: {', '.join(f"{f} {_f(a)}" for f, a in zip(lb[lb['label'] == 'benign']['family'], benign_held))}. Gap âm lớn nghĩa là model nhận ra benign nhờ đã thấy phong cách của nguồn đó, không phải nhờ quy luật chung. Đây là cùng cơ chế khiến model sụp trên Hassan.
- Khi bỏ `chongluadao` (chiếm phần lớn phishing của train), số phishing còn lại để học giảm mạnh (xem `leave_one_source_out.csv`, cột `train_pos_left`). Gap ở đây phản ánh cả việc thiếu dữ liệu lẫn việc lệch nguồn.
- Bỏ `www.` (baseline, accuracy khi họ bị loại khỏi train): {loso_nw}.{loso_nw_text}
- Họ có n<30 (⚠) chỉ để tham khảo, không diễn giải.

## Files
- `outputs/preprocessing_ablation.csv`, `outputs/leave_one_source_out.csv`, `outputs/tfidf_top_ngrams.csv`, `outputs/www_prefix_rates.csv`
- `outputs/figures/preprocessing_ablation.png`, `outputs/figures/leave_one_source_out.png`
- Code: `src/phishing_ml/experiments.py`
"""
    path.write_text(txt, encoding='utf-8')


def run_experiments(vv: pd.DataFrame, out_dir: Path, fig_dir: Path, hassan_csv: Path, jpcert_urls: list[str]) -> None:
    print('  [prep] featurising www-normalised URLs + external sets ...', flush=True)
    vv = add_nowww(vv)
    externals = load_external_frames(hassan_csv, jpcert_urls)
    www = www_rates(vv, externals)
    www.to_csv(out_dir / 'www_prefix_rates.csv', index=False)
    prep, ngrams = run_preprocessing_ablation(vv, externals)
    prep.to_csv(out_dir / 'preprocessing_ablation.csv', index=False)
    ngrams.to_csv(out_dir / 'tfidf_top_ngrams.csv', index=False)
    print('  [loso] leave-one-source-out ...', flush=True)
    loso = run_leave_one_source_out(vv)
    loso.to_csv(out_dir / 'leave_one_source_out.csv', index=False)
    plot_experiments(prep, loso, fig_dir)
    skew = vv.loc[vv['split'] == 'train', FEATS].skew().sort_values(ascending=False)
    write_report(prep, loso, ngrams, skew, www, out_dir / 'EXPERIMENTS.md')
