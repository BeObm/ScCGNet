
import os
import pickle
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy import special
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

# =============================================================================
# CONFIG -- edit these values
# =============================================================================
CONFIG = {
    "datasets": [
        {"name": "Young",
         "embedding": "data/embeddings/young_embedding.npy",
         "labels": "data/young_types.txt#True"},
        {"name": "QS-Diaphragm",
         "embedding": "data/embeddings/diaphragm_embedding.npy",
         "labels": "data/diaphragm_types.txt#True"},
        {"name": "Qx-spleen",
         "embedding": "data/embeddings/spleen_embedding.npy",
         "labels": "data/spleen_types.txt#True"},
        {"name": "Qx-Muscle",
         "embedding": "data/embeddings/qx_muscle_embedding.npy",
         "labels": "data/qx_muscle_types.txt#True"},
        {"name": "QS-Muscle",
         "embedding": "data/embeddings/qs_muscle_embedding.npy",
         "labels": "data/qs_muscle_types.txt#True"},
    ],
    "outdir": "dependence_results",

    # ---- speed ----
    "n_jobs": -1,          # parallel workers (-1 = all CPU cores, 1 = no parallelism)
    "resume": False,       # True: reuse cached per-dataset results in <outdir>/_cache

    "seeds": [0, 1, 2, 3, 4],
    "standardize": True,
    "kmeans_n_init": 10,
    "gmm_n_init": 5,
    "gmcm_n_init": 5,
    "gmm_reg_covar": 1e-4,
    "gmcm_max_iter": 200,
    "gmcm_tol": 1e-6,
    "gmcm_patience": 20,

    # ---- Analysis 2 ----
    "run_model_heldout": True,
    "heldout_seeds": None,

    "cv_folds": 5,
    "kde_defensive_eps": 0.05,
    "kde_chunk": 512,
    "min_cells_per_type": 30,
    "skew_threshold": 0.5,
    "kurt_threshold": 1.0,
    "rank_gap_threshold": 0.1,
    "n_qq_plots": 4,

    # ---- Analysis 3 ----
    "min_datasets_association": 5,

    # ---- Analysis 4 ----
    "run_synthetic": True,
    "synthetic": {
        "n_clusters": 5,
        "n_dims": 10,
        "n_per_cluster": 200,
        "separation": 3.0,
        "transforms": ["exp", "sinh"],
        "strengths": [0.0, 0.5, 1.0, 1.5, 2.0],
        "standardize": True,
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
}


HERE = os.path.dirname(os.path.abspath(__file__))
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _path(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def _standardize(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)



def _single_threaded(fn, *args):
    with threadpool_limits(limits=1):
        return fn(*args)


def _pmap(fn, tasks, n_jobs):
    tasks = list(tasks)
    if n_jobs == 1 or len(tasks) <= 1:
        return [fn(*t) for t in tasks]
    return Parallel(n_jobs=n_jobs)(delayed(_single_threaded)(fn, *t) for t in tasks)



# Loading

def load_embedding(path):
    path = _path(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        E = np.load(path)
    elif ext == ".npz":
        f = np.load(path)
        E = f[f.files[0]]
    elif ext in (".pt", ".pth"):
        import torch

        E = torch.load(path, map_location="cpu")
        E = E.detach().numpy() if hasattr(E, "detach") else np.asarray(E)
    elif ext in (".csv", ".tsv", ".txt"):
        E = pd.read_csv(path, sep=None, engine="python", header=None).values
        if not np.issubdtype(E.dtype, np.number):
            E = pd.read_csv(path, sep=None, engine="python", index_col=0).values
    else:
        raise ValueError(f"Unsupported embedding file: {path}")
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"Embedding must be 2-D (cells x dims), got shape {E.shape}")
    return E


CELLTYPE_KEYS = ["cluster", "cell_type1", "celltype", "cell_type", "CellType",
                 "cell_ontology_class", "free_annotation"]


def find_celltype_column(obs, keys=CELLTYPE_KEYS):
    for k in keys:
        if k in obs.columns:
            return k
    lower = {c.lower(): c for c in obs.columns}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    raise ValueError(f"No cell-type column found in adata.obs. Looked for {keys}; "
                     f"obs columns are {list(obs.columns)}.")


def labels_from_anndata(path, column=None):
    import anndata

    obs = anndata.read_h5ad(path, backed="r").obs
    if column in (None, "auto"):
        column = find_celltype_column(obs)
        print(f"Cell-type labels: adata.obs['{column}'] of {os.path.basename(path)}")
    elif column not in obs.columns:
        raise ValueError(f"Column '{column}' not in adata.obs; columns: {list(obs.columns)}")
    lab = np.array(["nan" if pd.isna(x) else str(x) for x in obs[column].tolist()])
    n_missing = (lab == "nan").sum()
    if n_missing:
        warnings.warn(f"{n_missing} cells have no label in obs['{column}'] (kept as 'nan').")
    return lab


def load_labels(spec):
    if spec.endswith(".h5ad") or ".h5ad::" in spec:
        path, key = spec.split("::", 1) if "::" in spec else (spec, "auto")
        lab = labels_from_anndata(_path(path), key)
    elif "::" in spec:
        import h5py

        path, key = spec.split("::", 1)
        with h5py.File(_path(path), "r") as f:
            lab = f[key][...]
    elif spec.endswith(".npy"):
        lab = np.load(_path(spec), allow_pickle=True)
    else:
        path, col = spec.rsplit("#", 1) if "#" in spec else (spec, None)
        df = pd.read_csv(_path(path), sep=None, engine="python")
        lab = df[col].values if col else df.iloc[:, -1].values
    lab = np.asarray(lab).ravel()
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in lab])



# Metrics

def cluster_acc(y, pred):
    _, yi = np.unique(y, return_inverse=True)
    _, pi = np.unique(pred, return_inverse=True)
    C = np.zeros((pi.max() + 1, yi.max() + 1), dtype=int)
    np.add.at(C, (pi, yi), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(y)


def scores(y, pred):
    return {"ACC": cluster_acc(y, pred),
            "NMI": normalized_mutual_info_score(y, pred),
            "ARI": adjusted_rand_score(y, pred)}



# Gaussian mixture copula model

def _log_gauss(Z, mean, cov):
    d = Z.shape[1]
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (Z - mean).T)
    return -0.5 * (d * np.log(2 * np.pi) + (sol ** 2).sum(0)) - np.log(np.diag(L)).sum()


def _joint_logpdf_components(Z, w, mu, cov):
    return np.column_stack([np.log(w[k]) + _log_gauss(Z, mu[k], cov[k]) for k in range(len(w))])


def _diag_sd(cov):
    return np.sqrt(np.diagonal(cov, axis1=1, axis2=2))


def _marginal_logpdf(Z, w, mu, cov):
    sd = _diag_sd(cov)
    t = (Z[:, :, None] - mu.T[None]) / sd.T[None]
    comp = np.log(w)[None, None, :] - 0.5 * t * t - np.log(sd.T)[None] - LOG_SQRT_2PI
    return _lse(comp, axis=2).sum(1)


def _standardize_params(w, mu, cov):
    m = (w[:, None] * mu).sum(0)
    var = (w[:, None] * (np.diagonal(cov, axis1=1, axis2=2) + mu ** 2)).sum(0) - m ** 2
    s = np.sqrt(np.maximum(var, 1e-12))
    return w, (mu - m) / s, cov / np.outer(s, s)[None, :, :]


def _invert_marginals(U, w, mu, cov, grid_size=4000):
    sd = _diag_sd(cov)
    lo = (mu - 9 * sd).min(0)
    hi = (mu + 9 * sd).max(0)
    T = lo[None, :] + (hi - lo)[None, :] * np.linspace(0.0, 1.0, grid_size)[:, None]
    F = (w[None, None, :] * special.ndtr((T[:, :, None] - mu.T[None]) / sd.T[None])).sum(2)
    F = np.maximum.accumulate(F, axis=0) + (np.arange(grid_size) * 1e-15)[:, None]
    Z = np.empty_like(U)
    for j in range(U.shape[1]):
        Z[:, j] = np.interp(U[:, j], F[:, j], T[:, j])
    return Z


def _copula_terms(U, w, mu, cov):
    Z = _invert_marginals(U, w, mu, cov)
    logp = _joint_logpdf_components(Z, w, mu, cov)
    log_c = _lse(logp, axis=1) - _marginal_logpdf(Z, w, mu, cov)
    return Z, logp, log_c


class GMCM:

    def __init__(self, n_components, max_iter=200, tol=1e-6, reg_covar=1e-4, random_state=None,
                 patience=20, n_init=1):
        self.K, self.max_iter, self.tol = n_components, max_iter, tol
        self.reg, self.random_state, self.patience = reg_covar, random_state, patience
        self.n_init = n_init

    def _fit_once(self, U, seed):
        n, d = U.shape
        gm = GaussianMixture(self.K, covariance_type="full", reg_covar=self.reg,
                             random_state=seed, n_init=1).fit(stats.norm.ppf(U))
        w, mu, cov = _standardize_params(gm.weights_.copy(), gm.means_.copy(), gm.covariances_.copy())
        ll_old, lab_old, stable, converged, n_iter = -np.inf, None, 0, False, 0
        eye = self.reg * np.eye(d)
        for it in range(self.max_iter):
            Z, logp, log_c = _copula_terms(U, w, mu, cov)
            ll = log_c.sum()
            lab = logp.argmax(1)
            n_iter = it + 1
            stable = stable + 1 if lab_old is not None and (lab == lab_old).all() else 0
            lab_old = lab
            if abs(ll - ll_old) <= self.tol * abs(ll) or stable >= self.patience:
                converged = True
                break
            ll_old = ll
            R = np.exp(logp - _lse(logp, axis=1)[:, None])
            Nk = R.sum(0) + 1e-10
            w = Nk / n
            mu = (R.T @ Z) / Nk[:, None]
            cov = np.stack([((R[:, k, None] * (Z - mu[k])).T @ (Z - mu[k])) / Nk[k] + eye
                            for k in range(self.K)])
            w, mu, cov = _standardize_params(w, mu, cov)
        _, logp, log_c = _copula_terms(U, w, mu, cov)
        return {"ll": log_c.sum(), "labels": logp.argmax(1), "converged": converged,
                "n_iter": n_iter, "params": (w, mu, cov)}

    def fit(self, X):
        U = stats.rankdata(X, axis=0) / (X.shape[0] + 1)
        base = 0 if self.random_state is None else self.random_state
        best = None
        for r in range(self.n_init):
            res = self._fit_once(U, base * 1000 + r)
            if best is None or res["ll"] > best["ll"]:
                best = res
        self.loglik_ = best["ll"]
        self.labels_ = best["labels"]
        self.converged_ = best["converged"]
        self.n_iter_ = best["n_iter"]
        self.params_ = best["params"]
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def copula_logpdf(self, U):
        return _copula_terms(U, *self.params_)[2]



# Marginal density estimates (defensive-mixture KDE)

def _kde_bw(x):
    sd = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
    s = min(sd, iqr) if iqr > 0 else sd
    return max(0.9 * s * len(x) ** (-0.2), 1e-3)


def _marginal_cdf_logpdf(train, test, h, eps, leave_one_out=False, want_logpdf=True, chunk=512):

    n, nt = len(train), len(test)
    m, s = train.mean(), max(train.std(), 1e-8)
    kcdf = np.empty(nt)
    log_kde = np.empty(nt) if want_logpdf else None
    for a in range(0, nt, chunk):
        t = (test[a:a + chunk, None] - train[None, :]) / h
        kcdf[a:a + chunk] = special.ndtr(t).mean(1)
        if want_logpdf:
            q = -0.5 * t * t
            ssum = np.exp(q).sum(1)
            with np.errstate(divide="ignore"):
                lk = np.log(ssum)
            bad = ~(ssum > 1e-250)
            if bad.any():
                lk[bad] = special.logsumexp(q[bad], axis=1)
            log_kde[a:a + chunk] = lk - LOG_SQRT_2PI - np.log(n) - np.log(h)
    if leave_one_out:
        kcdf = (n * kcdf - 0.5) / (n - 1)
    zt = (test - m) / s
    cdf = np.clip((1 - eps) * kcdf + eps * special.ndtr(zt), 1e-6, 1 - 1e-6)
    if not want_logpdf:
        return cdf, None
    lp_norm = -0.5 * zt * zt - LOG_SQRT_2PI - np.log(s)
    return cdf, np.logaddexp(np.log1p(-eps) + log_kde, np.log(eps) + lp_norm)


def _marginals(A, B, eps, need_train=True, chunk=512):
    """Train (leave-one-out) CDF values if need_train, test CDF values, and the
    test marginal log-density summed over dimensions."""
    U_tr = np.empty_like(A) if need_train else None
    U_te, lm = np.empty_like(B), 0.0
    for j in range(A.shape[1]):
        h = _kde_bw(A[:, j])
        if need_train:
            U_tr[:, j], _ = _marginal_cdf_logpdf(A[:, j], A[:, j], h, eps, leave_one_out=True,
                                                 want_logpdf=False, chunk=chunk)
        U_te[:, j], lp = _marginal_cdf_logpdf(A[:, j], B[:, j], h, eps, chunk=chunk)
        lm = lm + lp
    return U_tr, U_te, lm



# Analysis 1

def _cluster_one_seed(X, y, K, s, cfg, gmcm_labels=None):
    info = {"converged": True, "n_iter": None}
    if gmcm_labels is None:
        g = GMCM(K, cfg["gmcm_max_iter"], cfg["gmcm_tol"], cfg["gmm_reg_covar"], random_state=s,
                 patience=cfg["gmcm_patience"], n_init=cfg["gmcm_n_init"]).fit(X)
        gmcm_labels = g.labels_
        info = {"converged": g.converged_, "n_iter": g.n_iter_}
    preds = {
        "KMeans": KMeans(K, n_init=cfg["kmeans_n_init"], random_state=s).fit_predict(X),
        "GMM": GaussianMixture(K, covariance_type="full", reg_covar=cfg["gmm_reg_covar"],
                               n_init=cfg["gmm_n_init"], random_state=s).fit(X).predict(X),
        "GMCM": gmcm_labels,
    }
    return preds, info


def run_clustering(E, y, seeds, cfg, name, standardize=None, verbose=True, n_jobs=None,
                   gmcm_labels_given=None):
    K = len(np.unique(y))
    std = cfg["standardize"] if standardize is None else standardize
    X = _standardize(E) if std else E
    n_jobs = cfg["n_jobs"] if n_jobs is None else n_jobs
    given = gmcm_labels_given or {}
    out = _pmap(_cluster_one_seed, [(X, y, K, s, cfg, given.get(s)) for s in seeds], n_jobs)
    rows, gmcm_labels = [], {}
    for s, (preds, info) in zip(seeds, out):
        gmcm_labels[s] = preds["GMCM"]
        if not info["converged"] and verbose:
            print(f"  [{name}] seed {s}: GMCM stopped at max_iter={info['n_iter']} without converging")
        for m, p in preds.items():
            rows.append({"dataset": name, "method": m, "seed": s,
                         "n_clusters_found": len(np.unique(p)), **scores(y, p)})
        if verbose:
            print(f"  [{name}] seed {s}: " + ", ".join(
                f"{m} ARI={adjusted_rand_score(y, p):.3f}" for m, p in preds.items()))
    return pd.DataFrame(rows), gmcm_labels



# Analysis 2a: held-out GMM vs GMCM

def _heldout_one_fold(X, K, cfg, s, f, tr, te):
    A, B = X[tr], X[te]
    if cfg["standardize"]:
        m, sd = A.mean(0), A.std(0)
        sd = np.where(sd > 0, sd, 1)
        A, B = (A - m) / sd, (B - m) / sd
    gm = GaussianMixture(K, covariance_type="full", reg_covar=cfg["gmm_reg_covar"],
                         n_init=cfg["gmm_n_init"], random_state=s).fit(A)
    g = GMCM(K, cfg["gmcm_max_iter"], cfg["gmcm_tol"], cfg["gmm_reg_covar"], random_state=s,
             patience=cfg["gmcm_patience"], n_init=cfg["gmcm_n_init"]).fit(A)
    # only test-fold quantities are used, so the train-by-train KDE is skipped
    _, U_te, lm = _marginals(A, B, cfg["kde_defensive_eps"], need_train=False,
                             chunk=cfg["kde_chunk"])
    ll_gmm = gm.score_samples(B).mean()
    ll_gmcm = (g.copula_logpdf(U_te) + lm).mean()
    return {"seed": s, "fold": f, "heldout_ll_gmm": ll_gmm, "heldout_ll_gmcm": ll_gmcm,
            "gain_per_cell": ll_gmcm - ll_gmm, "gain_per_dim": (ll_gmcm - ll_gmm) / X.shape[1]}


def heldout_models(E, y, cfg, name):
    K = len(np.unique(y))
    X = E
    n = X.shape[0]
    seeds = cfg["heldout_seeds"] if cfg.get("heldout_seeds") else cfg["seeds"]
    tasks = []
    for s in seeds:
        idx = np.random.default_rng(s).permutation(n)
        parts = np.array_split(idx, cfg["cv_folds"])
        for f, te in enumerate(parts):
            tr = np.concatenate([parts[g] for g in range(cfg["cv_folds"]) if g != f])
            tasks.append((X, K, cfg, s, f, tr, te))
    t0 = time.time()
    rows = _pmap(_heldout_one_fold, tasks, cfg["n_jobs"])
    print(f"  [{name}] held-out models: {len(tasks)} folds done in {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", name)
    return df



# Analysis 2b: descriptive within-type statistics

def heldout_single_type(X, folds, seed, eps, reg=1e-6, chunk=512):
    n, d = X.shape
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, folds)
    ll_g, ll_c = [], []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        A, B = X[tr], X[te]
        cov = np.cov(A, rowvar=False) + reg * np.eye(d)
        ll_g.append(_log_gauss(B, A.mean(0), cov))
        U_tr, U_te, lm = _marginals(A, B, eps, need_train=True, chunk=chunk)
        Ztr, Zte = stats.norm.ppf(U_tr), stats.norm.ppf(U_te)
        R = np.corrcoef(Ztr, rowvar=False) + reg * np.eye(d)
        L = np.linalg.cholesky(R)
        q = (np.linalg.solve(L, Zte.T) ** 2).sum(0)
        ll_c.append(-np.log(np.diag(L)).sum() - 0.5 * (q - (Zte ** 2).sum(1)) + lm)
    return np.concatenate(ll_g).mean(), np.concatenate(ll_c).mean()


def _type_stats_one(Xt_full, t, cfg, name):
    keep = Xt_full.std(0) > 1e-8
    Xt = Xt_full[:, keep]
    n, d_t = Xt.shape
    if n < max(cfg["min_cells_per_type"], 3 * d_t):
        return None, [], f"  [{name}] type {t}: {n} cells for {d_t} usable dims, skipped"
    sk = stats.skew(Xt, axis=0)
    ku = stats.kurtosis(Xt, axis=0)
    pear = np.corrcoef(Xt, rowvar=False)
    spear = stats.spearmanr(Xt).statistic if d_t > 1 else np.eye(1)
    iu = np.triu_indices(d_t, 1)
    gap = np.abs(spear - pear)[iu]
    ev = np.maximum(np.linalg.eigvalsh(np.cov(Xt, rowvar=False)), 0.0)
    p = ev / ev.sum() if ev.sum() > 0 else np.full(d_t, 1.0 / d_t)
    eff_rank = np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())
    ll_g, ll_c = heldout_single_type(Xt, cfg["cv_folds"], 0, cfg["kde_defensive_eps"],
                                     chunk=cfg["kde_chunk"])
    row = {
        "dataset": name, "cell_type": t, "n_cells": n, "n_dims": d_t,
        "mean_abs_skew": np.abs(sk).mean(),
        "frac_dims_skewed": (np.abs(sk) > cfg["skew_threshold"]).mean(),
        "mean_abs_excess_kurtosis": np.abs(ku).mean(),
        "frac_dims_kurtosis": (np.abs(ku) > cfg["kurt_threshold"]).mean(),
        "rank_linear_discrepancy": gap.mean() if gap.size else np.nan,
        "frac_pairs_discrepancy": (gap > cfg["rank_gap_threshold"]).mean() if gap.size else np.nan,
        "effective_rank": eff_rank,
        "top_eig_over_mean": ev.max() / ev.mean() if ev.mean() > 0 else np.nan,
        "heldout_ll_gaussian": ll_g,
        "heldout_ll_gauss_copula": ll_c,
        "heldout_gain_per_dim": (ll_c - ll_g) / d_t,
    }
    dims = np.where(keep)[0]
    qq = [(np.abs(sk[j]), t, dims[j], Xt[:, j]) for j in np.argsort(-np.abs(sk))[:cfg["n_qq_plots"]]]
    return row, qq, None


def run_type_statistics(E, y, cfg, name, outdir):
    X = _standardize(E)
    types = sorted(np.unique(y))
    out = _pmap(_type_stats_one, [(X[y == t], t, cfg, name) for t in types], cfg["n_jobs"])
    rows, qq = [], []
    for row, q, msg in out:
        if msg:
            print(msg)
        if row is not None:
            rows.append(row)
        qq.extend(q)

    qq = sorted(qq, key=lambda r: -r[0])[:cfg["n_qq_plots"]]
    if qq:
        fig, axes = plt.subplots(1, len(qq), figsize=(3.2 * len(qq), 3.2))
        for ax, (s, t, j, x) in zip(np.atleast_1d(axes), qq):
            stats.probplot((x - x.mean()) / x.std(), dist="norm", plot=ax)
            ax.get_lines()[0].set(markersize=2, color="black")
            ax.get_lines()[1].set(color="grey")
            ax.set_title(f"type {t}, dim {j} (skew {stats.skew(x):.2f})", fontsize=9)
            ax.set_xlabel("Normal quantiles")
            ax.set_ylabel("Standardized values")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_qq.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(outdir, f"{name}_qq.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    return pd.DataFrame(rows)



# Analysis 3: exploratory cross-dataset association

def run_association(clus_summary, dataset_stats, outdir):
    piv = clus_summary.pivot(index="dataset", columns="method", values="ARI_mean")
    gain = (piv["GMCM"] - piv["GMM"]).rename("ARI_gain_GMCM_minus_GMM")
    tab = pd.concat([gain, dataset_stats], axis=1).dropna(axis=1, how="all").dropna()
    tab.to_csv(os.path.join(outdir, "association_table.csv"))
    stat_cols = [c for c in tab.columns if c != "ARI_gain_GMCM_minus_GMM"]
    rows = [{"statistic": c, "n_datasets": len(tab),
             "spearman_rho": stats.spearmanr(tab[c], tab["ARI_gain_GMCM_minus_GMM"]).statistic}
            for c in stat_cols]
    assoc = pd.DataFrame(rows)
    assoc.to_csv(os.path.join(outdir, "association_exploratory.csv"), index=False)

    fig, axes = plt.subplots(1, len(stat_cols), figsize=(3.8 * len(stat_cols), 3.4))
    for ax, c in zip(np.atleast_1d(axes), stat_cols):
        ax.scatter(tab[c], tab["ARI_gain_GMCM_minus_GMM"], color="black", s=18)
        for ds, r in tab.iterrows():
            ax.annotate(ds, (r[c], r["ARI_gain_GMCM_minus_GMM"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel(c.replace("_", " "))
        ax.set_ylabel("ARI gain (GMCM - GMM)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "association_exploratory.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "association_exploratory.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return assoc



# Analysis 4: controlled synthetic experiment (marginal misspecification)

def _make_gaussian_clusters(K, d, n, sep, rng):
    X, y = [], []
    for k in range(K):
        A = rng.normal(size=(d, d)) / np.sqrt(d)
        cov = A @ A.T + 0.5 * np.eye(d)
        X.append(rng.multivariate_normal(rng.normal(size=d) * sep, cov, n))
        y += [k] * n
    return _standardize(np.vstack(X)), np.array(y)


def _monotone_transform(X, a, kind, signs):
    if a == 0:
        return X.copy()
    if kind == "exp":
        return signs * np.expm1(a * signs * X) / a
    if kind == "sinh":
        return np.sinh(a * X) / a
    raise ValueError(f"Unknown transform: {kind}")


def _synthetic_one(kind, s, cfg):
    sc = cfg["synthetic"]
    rng = np.random.default_rng(s)
    X0, y = _make_gaussian_clusters(sc["n_clusters"], sc["n_dims"], sc["n_per_cluster"],
                                    sc["separation"], rng)
    signs = rng.choice([-1.0, 1.0], size=sc["n_dims"])
    rows, gmcm_by_strength = [], {}
    ref_ranks, ref_labels, n_refits = None, None, 0
    for a in sc["strengths"]:
        X = _monotone_transform(X0, a, kind, signs)
        if not np.isfinite(X).all():
            raise ValueError(f"{kind} strength {a} overflowed; use smaller strengths.")
        Xin = _standardize(X) if sc["standardize"] else X      # exactly what the GMCM sees
        ranks = stats.rankdata(Xin, axis=0)
        given = None
        if ref_ranks is not None and np.array_equal(ranks, ref_ranks):
            given = {s: ref_labels}                            # identical input -> identical labels
        res, glab = run_clustering(X, y, [s], cfg, f"synthetic_{kind}_{a}",
                                   standardize=sc["standardize"], verbose=False, n_jobs=1,
                                   gmcm_labels_given=given)
        if given is None:
            n_refits += 1
        if ref_ranks is None:
            ref_ranks, ref_labels = ranks, glab[s]
        res["transform"], res["strength"], res["synthetic_seed"] = kind, a, s
        rows.append(res)
        gmcm_by_strength[a] = glab[s]
    ref = gmcm_by_strength[sc["strengths"][0]]
    inv = {"transform": kind, "synthetic_seed": s,
           "gmcm_labels_identical_across_strengths":
               all((v == ref).all() for v in gmcm_by_strength.values()),
           "gmcm_fits": n_refits}
    return pd.concat(rows, ignore_index=True), inv


def run_synthetic(cfg, outdir):
    sc = cfg["synthetic"]
    tasks = [(kind, s, cfg) for kind in sc["transforms"] for s in sc["seeds"]]
    t0 = time.time()
    out = _pmap(_synthetic_one, tasks, cfg["n_jobs"])
    print(f"  synthetic: {len(tasks)} datasets done in {time.time() - t0:.0f} s")
    runs = pd.concat([r for r, _ in out], ignore_index=True)
    runs.to_csv(os.path.join(outdir, "synthetic_per_seed.csv"), index=False)
    inv = pd.DataFrame([i for _, i in out])
    inv.to_csv(os.path.join(outdir, "synthetic_gmcm_invariance.csv"), index=False)
    summ = runs.groupby(["transform", "strength", "method"]).agg(
        ARI_mean=("ARI", "mean"), ARI_sd=("ARI", "std"),
        NMI_mean=("NMI", "mean"), ACC_mean=("ACC", "mean"),
        n_datasets=("ARI", "count")).reset_index()
    summ.to_csv(os.path.join(outdir, "synthetic_summary.csv"), index=False)

    kinds = sc["transforms"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.2), sharey=True)
    styles = {"KMeans": ("s", ":"), "GMM": ("o", "--"), "GMCM": ("^", "-")}
    label = {"exp": "exp (skewed marginals)", "sinh": "sinh (heavy-tailed marginals)"}
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        for m, (mk, ls) in styles.items():
            d = summ[(summ["method"] == m) & (summ["transform"] == kind)]
            ax.errorbar(d.strength, d.ARI_mean, yerr=d.ARI_sd, marker=mk, ls=ls, color="black",
                        capsize=2, ms=4, lw=1, label=m)
        ax.set_xlabel(f"Transform strength, {label.get(kind, kind)}")
        ax.set_ylim(-0.05, 1.05)
    np.atleast_1d(axes)[0].set_ylabel("ARI (mean ± sd)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "synthetic_ari.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "synthetic_ari.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summ, inv



# Main

def _process_dataset(ds, cfg, outdir):
    name = ds["name"]
    E = load_embedding(ds["embedding"])
    y = load_labels(ds["labels"])
    if len(y) != E.shape[0]:
        raise ValueError(f"{name}: {E.shape[0]} embedding rows but {len(y)} labels")
    print(f"Embedding: {E.shape[0]} cells x {E.shape[1]} dims; {len(np.unique(y))} labels")
    print("NOTE: embedding rows must be in the same cell order as the labels (not checkable here).")

    t0 = time.time()
    print("Analysis 1: cluster models on the fixed embedding")
    runs = run_clustering(E, y, cfg["seeds"], cfg, name)[0]
    print(f"  ({time.time() - t0:.0f} s)")
    models = None
    if cfg["run_model_heldout"]:
        print("Analysis 2a: held-out GMM vs GMCM")
        models = heldout_models(E, y, cfg, name)
    t0 = time.time()
    print("Analysis 2b: within-type statistics")
    types = run_type_statistics(E, y, cfg, name, outdir)
    print(f"  ({time.time() - t0:.0f} s)")
    return runs, models, types


def main(cfg=CONFIG):
    outdir = _path(cfg["outdir"])
    os.makedirs(outdir, exist_ok=True)
    cache_dir = os.path.join(outdir, "_cache")
    if cfg.get("resume"):
        os.makedirs(cache_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    pd.set_option("display.width", 200)
    t_start = time.time()

    all_runs, all_types, all_models = [], [], []
    for ds in cfg["datasets"]:
        name = ds["name"]
        print(f"\n############## {name} ##############")
        cache = os.path.join(cache_dir, f"{name}.pkl")
        if cfg.get("resume") and os.path.exists(cache):
            with open(cache, "rb") as fh:
                runs, models, types = pickle.load(fh)
            print(f"Loaded from cache: {cache}")
        else:
            runs, models, types = _process_dataset(ds, cfg, outdir)
            if cfg.get("resume"):
                with open(cache, "wb") as fh:
                    pickle.dump((runs, models, types), fh)
        all_runs.append(runs)
        if models is not None:
            all_models.append(models)
        all_types.append(types)

    if all_runs:
        runs = pd.concat(all_runs, ignore_index=True)
        runs.to_csv(os.path.join(outdir, "clustering_per_seed.csv"), index=False)
        summ = runs.groupby(["dataset", "method"]).agg(
            ACC_mean=("ACC", "mean"), ACC_sd=("ACC", "std"),
            NMI_mean=("NMI", "mean"), NMI_sd=("NMI", "std"),
            ARI_mean=("ARI", "mean"), ARI_sd=("ARI", "std"),
            n_seeds=("seed", "count")).reset_index()
        summ.to_csv(os.path.join(outdir, "clustering_summary.csv"), index=False)
        print("\n=== Analysis 1: clustering (mean over seeds) ===")
        print(summ[["dataset", "method", "ACC_mean", "ACC_sd", "NMI_mean", "ARI_mean", "ARI_sd"]]
              .to_string(index=False, float_format="%.3f"))

        dstats = []
        if all_models:
            models = pd.concat(all_models, ignore_index=True)
            models.to_csv(os.path.join(outdir, "heldout_models_per_fold.csv"), index=False)
            msum = models.groupby("dataset").agg(
                heldout_ll_gmm=("heldout_ll_gmm", "mean"),
                heldout_ll_gmcm=("heldout_ll_gmcm", "mean"),
                gain_per_cell_mean=("gain_per_cell", "mean"),
                gain_per_cell_sd=("gain_per_cell", "std"),
                heldout_gain_per_dim=("gain_per_dim", "mean"),
                frac_folds_gmcm_better=("gain_per_cell", lambda g: (g > 0).mean()))
            msum.to_csv(os.path.join(outdir, "heldout_models_summary.csv"))
            print("\n=== Analysis 2a: held-out log-density per cell, GMCM (+ defensive-mixture KDE "
                  "marginals) vs GMM ===")
            print(msum.to_string(float_format="%.3f"))
            dstats.append(msum[["heldout_gain_per_dim"]].rename(
                columns={"heldout_gain_per_dim": "model_heldout_gain_per_dim"}))

        types = pd.concat(all_types, ignore_index=True)
        types.to_csv(os.path.join(outdir, "type_statistics.csv"), index=False)
        if len(types):
            tsum = types.groupby("dataset").agg(
                n_types=("cell_type", "count"),
                median_mean_abs_skew=("mean_abs_skew", "median"),
                median_mean_abs_excess_kurtosis=("mean_abs_excess_kurtosis", "median"),
                median_rank_linear_discrepancy=("rank_linear_discrepancy", "median"),
                median_effective_rank=("effective_rank", "median"),
                median_type_heldout_gain_per_dim=("heldout_gain_per_dim", "median"))
            tsum.to_csv(os.path.join(outdir, "type_statistics_summary.csv"))
            print("\n=== Analysis 2b: within-type statistics (median over types, descriptive) ===")
            print(tsum.to_string(float_format="%.3f"))
            dstats.append(tsum[["median_mean_abs_skew", "median_mean_abs_excess_kurtosis",
                                "median_rank_linear_discrepancy"]])

        n_ds = len(cfg["datasets"])
        if n_ds >= cfg["min_datasets_association"] and dstats:
            print("\n=== Analysis 3: exploratory association with the GMCM - GMM ARI gain ===")
            print(run_association(summ, pd.concat(dstats, axis=1), outdir)
                  .to_string(index=False, float_format="%.3f"))
        else:
            print(f"\nAnalysis 3 skipped: {n_ds} datasets, needs at least "
                  f"{cfg['min_datasets_association']}.")

    if cfg.get("run_synthetic", False):
        print("\n=== Analysis 4: synthetic experiment (marginal misspecification) ===")
        syn, inv = run_synthetic(cfg, outdir)
        for kind in cfg["synthetic"]["transforms"]:
            print(f"\n{kind}: mean ARI")
            print(syn[syn["transform"] == kind].pivot(index="strength", columns="method", values="ARI_mean")
                  .to_string(float_format="%.3f"))
        ok = inv["gmcm_labels_identical_across_strengths"]
        print(f"\nGMCM labels identical across strengths: {ok.sum()} of {len(ok)} synthetic datasets")
    print(f"\nResults written to: {outdir}  (total {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()