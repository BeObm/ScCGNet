import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

import marker_genes as sa


DATASETS = [
    {
        "name": "Young",
        "expr": "data/Young.h5ad",
        "labels": {
            "truth": "data/young_types.txt#True",
            "scCGNet": "data/young_types.txt#Pred",
        },
    },
    {
        "name": "Diaphragm",
        "expr": "data/Quake_Smart-seq2_Diaphragm.h5ad",
        "labels": {
            "truth": "data/diaphragm_types.txt#True",
            "scCGNet": "data/diaphragm_types.txt#Pred",
        },
    },

{
        "name": "Qx-Muscle",
        "expr": "data/Quake_10x_Limb_Muscle.h5ad",
        "labels": {
            "truth": "data/qx_muscle_types.txt#True",
            "scCGNet": "data/qx_muscle_types.txt#Pred",
        },
    },

{
        "name": "QS-Muscle",
        "expr": "data/Quake_Smart-seq2_Limb_Muscle.h5ad",
        "labels": {
            "truth": "data/qs_muscle_types.txt#True",
            "scCGNet": "data/qs_muscle_types.txt#Pred",
        },
    }

]


CONFIG = {
    "outdir_root": "internal_metrics",
    "spaces": ["pca"],
    "n_top_genes": 2000,
    "n_pcs": 50,
    "silhouette_sample": 5000,
    "silhouette_repeats": 5,
    "seed": 0,
    "x_key": None, "gene_key": None, "gene_col": "auto",
}





def pca_space(X, n_top_genes, n_pcs, seed):

    Xlog = sa.log_normalize(X).tocsc()

    mean = np.asarray(Xlog.mean(0)).ravel()
    sq = np.asarray(Xlog.multiply(Xlog).mean(0)).ravel()
    var = sq - mean ** 2
    top = np.argsort(-var)[:min(n_top_genes, Xlog.shape[1])]
    D = Xlog[:, top].toarray()
    sd = D.std(0)
    D = np.clip((D - D.mean(0)) / np.where(sd > 0, sd, 1), -10, 10)
    n_comp = min(n_pcs, D.shape[0] - 1, D.shape[1])
    return PCA(n_components=n_comp, random_state=seed).fit_transform(D)


def score(Z, labels, cfg, rng):
    """Metrics for one clustering in one space."""
    labels = np.asarray(labels)
    groups, counts = np.unique(labels, return_counts=True)
    rows = []
    k = len(groups)
    if k < 2:
        return [{"n_clusters": k, "silhouette": np.nan, "calinski_harabasz": np.nan,
                 "davies_bouldin": np.nan, "repeat": 0}]
    ch = calinski_harabasz_score(Z, labels)
    db = davies_bouldin_score(Z, labels)
    keep = np.isin(labels, groups[counts >= 2])

    Zs, ls = Z[keep], labels[keep]
    n = len(ls)
    for r in range(cfg["silhouette_repeats"] if n > cfg["silhouette_sample"] else 1):
        idx = rng.choice(n, cfg["silhouette_sample"], replace=False) if n > cfg["silhouette_sample"] \
            else np.arange(n)
        sil = silhouette_score(Zs[idx], ls[idx]) if len(np.unique(ls[idx])) > 1 else np.nan
        rows.append({"n_clusters": k, "silhouette": sil, "calinski_harabasz": ch,
                     "davies_bouldin": db, "repeat": r})
    return rows


def run_dataset(ds, cfg):
    name = ds["name"]
    print(f"\n############## {name} ##############")
    clusterings = {}
    for lname, spec in ds["labels"].items():
        path, col = spec.rsplit("#", 1) if "#" in spec else (spec, None)
        clusterings[lname] = sa.load_labels(sa._path(path), col)
    sizes = {len(v) for v in clusterings.values()}
    if len(sizes) != 1:
        raise ValueError(f"{name}: label sources have different numbers of cells: {sizes}")
    n_cells = sizes.pop()

    spaces = {}
    if "pca" in cfg["spaces"]:
        X, _ = sa.load_expression(sa._path(ds["expr"]), n_cells, ds.get("x_key", cfg["x_key"]),
                                  ds.get("gene_key", cfg["gene_key"]),
                                  ds.get("gene_col", cfg["gene_col"]))
        spaces["pca"] = pca_space(X, cfg["n_top_genes"], cfg["n_pcs"], cfg["seed"])
        print(f"PCA space: {spaces['pca'].shape[1]} components from the "
              f"{min(cfg['n_top_genes'], X.shape[1])} most variable genes")
    if "embedding" in cfg["spaces"] and ds.get("embedding"):
        E = np.load(sa._path(ds["embedding"]))
        if E.shape[0] != n_cells:
            raise ValueError(f"{name}: embedding has {E.shape[0]} rows, labels have {n_cells}")
        spaces["embedding"] = E
        print(f"Embedding space: {E.shape[1]} dimensions")

    rows = []
    rng = np.random.default_rng(cfg["seed"])
    for sname, Z in spaces.items():
        for lname, lab in clusterings.items():
            for r in score(Z, lab, cfg, rng):
                rows.append({"dataset": name, "space": sname, "clustering": lname, **r})
    return pd.DataFrame(rows)


def main(datasets=DATASETS, cfg=CONFIG):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = sa._path(cfg["outdir_root"])
    os.makedirs(out, exist_ok=True)
    runs = pd.concat([run_dataset(ds, cfg) for ds in datasets], ignore_index=True)
    runs.to_csv(os.path.join(out, "internal_metrics_per_run.csv"), index=False)
    summ = runs.groupby(["dataset", "space", "clustering"], sort=False).agg(
        n_clusters=("n_clusters", "first"),
        silhouette_mean=("silhouette", "mean"), silhouette_sd=("silhouette", "std"),
        calinski_harabasz=("calinski_harabasz", "first"),
        davies_bouldin=("davies_bouldin", "first"),
        n_silhouette_subsamples=("repeat", "count")).reset_index()
    summ.to_csv(os.path.join(out, "internal_metrics_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + summ.to_string(index=False, float_format="%.3f"))
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()