
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats


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
    "outdir_root": "Gene_markers_results",

    # ---- loading ----
    "x_key": None,
    "gene_key": None,
    "gene_col": "auto",

    # ---- marker genes ----
    "top_k": 3,
    "padj": 0.05,
    "min_log2fc": 0.25,
    "min_pct": 0.1,
    "min_cells": 3,
    "unique": True,
    "scale_per_gene": True,
    "plot_title": False,
    "marker_db": None,
    "db_celltype_col": "cell_name",
    "db_gene_col": "Symbol",
    "db_top_n": 50,
}
# =============================================================================



HERE = os.path.dirname(os.path.abspath(__file__))


def _path(p):
    return p if p is None or os.path.isabs(p) else os.path.join(HERE, p)


def _decode(arr):
    return np.array([a.decode() if isinstance(a, bytes) else str(a) for a in arr])


def _describe_h5(f):
    import h5py

    lines = []
    f.visititems(lambda k, o: lines.append(
        f"  {k}: {o.shape} {o.dtype}" if isinstance(o, h5py.Dataset) else f"  {k}/ (group)"))
    return "\n".join(lines)


GENE_COL_CANDIDATES = [
    "gene_symbols", "gene_symbol", "gene_names", "gene_name", "symbols", "symbol",
    "genes", "gene", "feature_name", "feature_names", "features", "gene_short_name",
    "hgnc_symbol", "mgi_symbol", "external_gene_name", "name", "names",
    "gene_ids", "gene_id", "ensembl_id", "ensembl_ids",
]


def _looks_like_symbols(values):
    v = pd.Series(values).astype(str).head(2000)
    if len(v) == 0:
        return False
    numeric = v.str.fullmatch(r"-?\d+(\.\d+)?").mean()
    empty = v.isin(["", "nan", "None"]).mean()
    return numeric < 0.05 and empty < 0.05 and v.nunique() > 0.9 * len(v)


def _values_summary(values, k=3):
    v = pd.Series(np.asarray(values).ravel()).astype(str)
    return ", ".join(v.head(k)) + (", ..." if len(v) > k else "")


def _symbol_sources(ad):

    n = ad.n_vars
    out = [("adata.var_names", np.asarray(ad.var_names))]
    lower = {c.lower(): c for c in ad.var.columns}
    ordered = [lower[c.lower()] for c in GENE_COL_CANDIDATES if c.lower() in lower]
    ordered += [c for c in ad.var.columns if c not in ordered]
    out += [(f"adata.var['{c}']", ad.var[c].values) for c in ordered]
    raw = getattr(ad, "raw", None)
    if raw is not None and raw.n_vars == n:
        out.append(("adata.raw.var_names", np.asarray(raw.var_names)))
        out += [(f"adata.raw.var['{c}']", raw.var[c].values) for c in raw.var.columns]
    for key, val in getattr(ad, "uns", {}).items():
        try:
            arr = np.asarray(val).ravel()
        except Exception:
            continue
        if arr.ndim == 1 and len(arr) == n:
            out.append((f"adata.uns['{key}']", arr))
    return [(lab, v) for lab, v in out if len(v) == n]


def _describe_gene_sources(ad):
    lines = [f"  {lab}: dtype={pd.Series(np.asarray(v).ravel()).dtype}, values: {_values_summary(v)}"
             for lab, v in _symbol_sources(ad)]
    return "\n".join(lines) if lines else "  (no source with one value per gene)"


def _anndata_counts_genes(ad, gene_col="auto", candidates=GENE_COL_CANDIDATES):

    X = ad.layers["counts"] if "counts" in ad.layers else ad.X
    var_names = np.asarray(ad.var_names).astype(str)
    if gene_col is None:
        return X, var_names
    if gene_col != "auto":
        if gene_col not in ad.var.columns:
            raise ValueError(f"gene_col '{gene_col}' not in adata.var; columns: {list(ad.var.columns)}")
        vals = np.asarray([str(x) for x in ad.var[gene_col].tolist()])
        if not _looks_like_symbols(vals):
            warnings.warn(f"adata.var['{gene_col}'] does not look like gene symbols "
                          f"(values: {_values_summary(vals)}).")
        return X, vals

    sources = _symbol_sources(ad)
    def _is_ensembl(v):
        return pd.Series(np.asarray(v).ravel()).astype(str).head(200).str.startswith("ENS").mean() > 0.5
    symbols = [(lab, v) for lab, v in sources if _looks_like_symbols(v)]
    for lab, v in sorted(symbols, key=lambda s: _is_ensembl(s[1])):  # stable: preference order kept
        note = " (Ensembl IDs, no symbols found)" if _is_ensembl(v) else ""
        print(f"Gene names: {lab}{note}, e.g. {_values_summary(v)}")
        return X, np.asarray([str(x) for x in np.asarray(v).ravel()])
    warnings.warn(
        "No gene-symbol source found in the .h5ad file; marker genes will be shown as "
        f"{_values_summary(var_names)} and will not be interpretable. Sources checked "
        "(one value per gene):\n" + _describe_gene_sources(ad) +
        "\nIf the names are in a separate file, set 'gene_file' for this dataset "
        "(text file, one gene name per line, in the order of the matrix columns).")
    return X, var_names


def _read_h5(path, x_key, gene_key, n_cells, gene_col="auto"):
    import h5py

    with h5py.File(path, "r") as f:
        enc = f.attrs.get("encoding-type", b"")
        is_anndata = (enc in ("anndata", b"anndata")) or (
            "X" in f and "var" in f and "_index" in f["var"].attrs)
        if is_anndata:  # AnnData file saved with a .h5 extension
            import anndata

            X, genes = _anndata_counts_genes(anndata.read_h5ad(path), gene_col)
            return sp.csr_matrix(X), genes

        dense, sparse = {}, {}

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and {"data", "indices", "indptr"} <= set(obj.keys()):
                sparse[name] = obj
            elif isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.dtype.kind in "iuf":
                dense[name] = obj

        f.visititems(visit)

        if x_key is None:
            sizes = {k: np.prod(v.shape) for k, v in dense.items()}
            sizes.update({k: v["data"].shape[0] * 1e6 for k, v in sparse.items()})  # prefer sparse groups
            if not sizes:
                raise ValueError("No count matrix found in the .h5 file. Contents:\n" + _describe_h5(f))
            x_key = max(sizes, key=sizes.get)
        obj = f[x_key]
        if isinstance(obj, h5py.Group):
            shape = None
            for s in ("shape", "h5sparse_shape"):
                if s in obj:
                    shape = tuple(int(v) for v in obj[s][...])
                elif s in obj.attrs:
                    shape = tuple(int(v) for v in obj.attrs[s])
                if shape:
                    break
            if shape is None:
                raise ValueError(f"Sparse group '{x_key}' has no shape. Contents:\n" + _describe_h5(f))
            data, ind, ptr = obj["data"][...], obj["indices"][...], obj["indptr"][...]
            enc = obj.attrs.get("encoding-type", b"")
            enc = enc.decode() if isinstance(enc, bytes) else str(enc)
            if enc == "csc_matrix" or (enc != "csr_matrix" and len(ptr) == shape[1] + 1):
                X = sp.csc_matrix((data, ind, ptr), shape=shape).tocsr()
            else:
                X = sp.csr_matrix((data, ind, ptr), shape=shape)
        else:
            X = sp.csr_matrix(obj[...])
        print(f"Count matrix: '{x_key}' with shape {X.shape}")

        # --- orientation: rows must be cells ---
        if X.shape[0] != n_cells and X.shape[1] == n_cells:
            X = X.T.tocsr()
            print("  transposed to cells x genes")
        if X.shape[0] != n_cells:
            raise ValueError(
                f"Matrix '{x_key}' has shape {X.shape}, but the labels file has {n_cells} cells.\n"
                "Contents of the .h5 file:\n" + _describe_h5(f))

        # --- gene names ---
        if gene_key is None:
            cands = []

            def visit_names(name, obj):
                if (isinstance(obj, h5py.Dataset) and obj.ndim == 1
                        and obj.shape[0] == X.shape[1] and obj.dtype.kind in "SOU"):
                    low = name.lower()
                    score = ("gene" in low) * 2 + ("var" in low) + ("symbol" in low) * 3 \
                        - ("id" in low.split("/")[-1]) - 5 * any(w in low for w in ("cell", "obs", "barcode"))
                    cands.append((score, name))

            f.visititems(visit_names)
            cands = [(sc + 10 * _looks_like_symbols(_decode(f[nm][:2000])), nm) for sc, nm in cands]
            if not cands:
                raise ValueError(
                    f"No gene-name dataset of length {X.shape[1]} found. Set CONFIG['gene_key'].\n"
                    "Contents of the .h5 file:\n" + _describe_h5(f))
            gene_key = max(cands)[1]
        genes = _decode(f[gene_key][...])
        print(f"Gene names: '{gene_key}' (e.g. {', '.join(genes[:5])})")
    return X, genes


def load_expression(path, n_cells, x_key=None, gene_key=None, gene_col="auto"):
    """Return (X, genes) with X of shape (cells, genes), raw counts for all genes."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        X, genes = _read_h5(path, x_key, gene_key, n_cells, gene_col)
    elif ext == ".h5ad":
        import anndata

        X, genes = _anndata_counts_genes(anndata.read_h5ad(path), gene_col)
    elif ext in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="," if ext == ".csv" else "\t", index_col=0)
        if df.shape[0] != n_cells and df.shape[1] == n_cells:
            df = df.T
        X, genes = df.values, np.asarray(df.columns.astype(str))
    else:
        raise ValueError(f"Unsupported expression file: {path}")
    X = sp.csr_matrix(X, dtype=np.float32)
    if X.shape[0] != n_cells:
        raise ValueError(f"Expression has {X.shape[0]} cells but labels have {n_cells}.")
    return X, genes


CELLTYPE_KEYS = ["cluster", "cell_type1", "celltype", "cell_type", "CellType",
                 "cell_ontology_class", "free_annotation"]


def find_celltype_column(obs, keys=CELLTYPE_KEYS):
    """First column of `obs` named in `keys` (exact match first, then ignoring case)."""
    for k in keys:
        if k in obs.columns:
            return k
    lower = {c.lower(): c for c in obs.columns}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    raise ValueError(f"No cell-type column found in adata.obs. Looked for {keys}; "
                     f"obs columns are {list(obs.columns)}. Add the right name to CELLTYPE_KEYS "
                     "or give it explicitly (e.g. 'file.h5ad#column').")


def labels_from_anndata(path, column=None):
    """Cell-type labels from adata.obs. column None or 'auto' searches CELLTYPE_KEYS."""
    import anndata

    obs = anndata.read_h5ad(path, backed="r").obs
    if column in (None, "auto"):
        column = find_celltype_column(obs)
        print(f"Cell-type labels: adata.obs['{column}'] of {os.path.basename(path)}")
    elif column not in obs.columns:
        raise ValueError(f"Column '{column}' not in adata.obs; columns: {list(obs.columns)}")
    lab = np.array(["nan" if pd.isna(x) else str(x) for x in obs[column].tolist()])
    n_missing = np.isin(lab, ["nan", "NaN", "None", ""]).sum()
    if n_missing:
        warnings.warn(f"{n_missing} cells have no label in obs['{column}'] (kept as the label 'nan').")
    return lab


def load_labels(path, column=None):

    if path.endswith(".h5ad"):
        return labels_from_anndata(path, column)
    if path.endswith(".npy"):
        lab = np.load(path, allow_pickle=True)
        if lab.ndim == 2:  # soft assignments -> hard labels
            lab = lab.argmax(1)
    elif column is not None:
        sep = "\t" if path.endswith(".tsv") else ","
        lab = pd.read_csv(path, sep=sep)[column].values
    else:
        lab = np.loadtxt(path, dtype=str, ndmin=1)
    return np.asarray(lab).astype(str).ravel()


def sorted_groups(labels):
    groups = [str(g) for g in np.unique(labels)]
    try:
        return sorted(groups, key=lambda g: float(g))
    except ValueError:
        return sorted(groups)



def log_normalize(X, target_sum=1e4):
    if X.data.size and X.data.min() < 0:
        raise ValueError("Negative values found: pass raw counts, not scaled data.")
    if X.data.size and not np.allclose(X.data[:10000], np.round(X.data[:10000])):
        warnings.warn("Input is not integer-valued; check that it contains raw counts.")
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1.0
    Xn = sp.diags((target_sum / lib).astype(np.float32)) @ X
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsc()


def wilcoxon_one_vs_rest(Xlog, labels, groups, gene_names, chunk=1000):
    """One-vs-rest Wilcoxon rank-sum test for every group and gene."""
    n_cells, n_genes = Xlog.shape
    masks = {g: labels == g for g in groups}
    rows = {g: [] for g in groups}

    for start in range(0, n_genes, chunk):
        stop = min(start + chunk, n_genes)
        D = Xlog[:, start:stop].toarray()
        ranks = stats.rankdata(D, axis=0)
        tie = np.array([stats.tiecorrect(ranks[:, j]) for j in range(D.shape[1])])
        expr = np.expm1(D)
        detected = D > 0

        for g in groups:
            m = masks[g]
            n1 = m.sum()
            n2 = n_cells - n1
            U = ranks[m].sum(0) - n1 * (n1 + 1) / 2.0
            sigma = np.sqrt(n1 * n2 * (n_cells + 1) / 12.0 * tie)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(sigma > 0, (U - n1 * n2 / 2.0) / sigma, 0.0)
            mean_in, mean_out = expr[m].mean(0), expr[~m].mean(0)
            rows[g].append(
                pd.DataFrame(
                    {
                        "group": g,
                        "gene": gene_names[start:stop],
                        "z": z,
                        "pval": 2 * stats.norm.sf(np.abs(z)),
                        # log2 ratio of mean normalized expression (not log-scale means)
                        "log2fc": np.log2((mean_in + 1e-9) / (mean_out + 1e-9)),
                        "pct_in": detected[m].mean(0),
                        "pct_out": detected[~m].mean(0),
                    }
                )
            )

    out = []
    for g in groups:
        df = pd.concat(rows[g], ignore_index=True)
        df["padj"] = stats.false_discovery_control(df["pval"].values, method="bh")
        out.append(df.sort_values("z", ascending=False))
    return pd.concat(out, ignore_index=True)


def select_markers(de, groups, k=3, padj_max=0.05, min_log2fc=0.25, min_pct=0.1, unique=False):
    """Top-k genes per group ranked by z after filtering.

    With unique=True, each gene is kept only for the group where its z-score
    is highest, so no gene is shown for two clusters.
    """
    cand = de[
        (de.padj < padj_max) & (de.log2fc > min_log2fc) & (de.pct_in >= min_pct) & (de.z > 0)
    ]
    if unique:
        cand = cand.loc[cand.groupby("gene")["z"].idxmax()]
    top = (
        cand.sort_values("z", ascending=False)
        .groupby("group", sort=False)
        .head(k)
    )
    order = {g: i for i, g in enumerate(groups)}
    top = top.assign(_o=top.group.map(order)).sort_values(["_o", "z"], ascending=[True, False])
    return top.drop(columns="_o").reset_index(drop=True)


def specificity_summary(markers, groups, k, name):
    """Descriptive statistics of the selected markers for one clustering."""
    per_group = markers.groupby("group")
    return {
        "clustering": name,
        "n_clusters": len(groups),
        "clusters_with_fewer_than_k_markers": int(
            sum(per_group.size().reindex(groups, fill_value=0) < k)
        ),
        "mean_pct_in": markers.pct_in.mean(),
        "mean_pct_out": markers.pct_out.mean(),
        "mean_pct_diff": (markers.pct_in - markers.pct_out).mean(),
        "mean_log2fc": markers.log2fc.mean(),
        "median_log2fc": markers.log2fc.median(),
        "n_distinct_markers": markers.gene.nunique(),
        "n_markers": len(markers),
    }


def dotplot(Xlog, labels, groups, markers, gene_names, out_path, title=None, scale_per_gene=True,
            ylabel="Predicted cluster"):
    genes = list(markers.gene)
    gene_group = list(markers.group)
    idx = pd.Index(gene_names).get_indexer(genes)
    D = Xlog[:, idx].toarray()

    mean = np.vstack([D[labels == g].mean(0) for g in groups])
    frac = np.vstack([(D[labels == g] > 0).mean(0) for g in groups])
    if scale_per_gene:  # scale each gene to [0, 1] across groups
        lo, hi = mean.min(0), mean.max(0)
        mean = (mean - lo) / np.where(hi > lo, hi - lo, 1)

    nG, nM = len(groups), len(genes)
    fig, ax = plt.subplots(figsize=(0.32 * nM + 2.5, 0.32 * nG + 1.8))
    xx, yy = np.meshgrid(np.arange(nM), np.arange(nG))
    sc = ax.scatter(
        xx.ravel(), yy.ravel(), s=frac.ravel() * 120, c=mean.ravel(),
        cmap="Reds", vmin=0, vmax=1 if scale_per_gene else None,
        edgecolors="grey", linewidths=0.3,
    )
    ax.set_xticks(range(nM))
    ax.set_xticklabels(genes, rotation=90, fontsize=8)
    ax.set_yticks(range(nG))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlim(-0.6, nM - 0.4)
    ax.set_ylim(nG - 0.4, -0.6)
    ax.set_ylabel(ylabel)

    # label blocks of genes by the cluster they were selected for
    start = 0
    for i in range(1, nM + 1):
        if i == nM or gene_group[i] != gene_group[start]:
            ax.annotate(
                gene_group[start], xy=((start + i - 1) / 2, -0.9), ha="center",
                annotation_clip=False, fontsize=8,
            )
            ax.plot([start - 0.3, i - 0.7], [-0.75, -0.75], color="black", lw=0.6, clip_on=False)
            start = i

    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Mean expression\n(scaled per gene)" if scale_per_gene else "Mean log expression", fontsize=8)
    handles = [ax.scatter([], [], s=f * 120, c="lightgrey", edgecolors="grey", label=f"{int(f * 100)}%")
               for f in (0.2, 0.6, 1.0)]
    # place the size legend just right of the colorbar label, independent of plot width
    cb.ax.legend(handles=handles, title="Fraction\nof cells", fontsize=7, title_fontsize=7,
                 loc="upper left", bbox_to_anchor=(5.5, 1.0), frameon=False)
    if title:
        ax.set_title(title, pad=22, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def marker_db_overlap(de, groups, db, background, top_n=50, min_set=3, report=3):

    bg = {g.upper() for g in background}
    sets = {
        ct: {g.upper() for g in grp["gene"]} & bg
        for ct, grp in db.groupby("cell_type")
    }
    sets = {ct: s for ct, s in sets.items() if len(s) >= min_set}
    M = len(bg)
    rows = []
    for g in groups:
        sub = de[(de.group == g) & (de.log2fc > 0)].sort_values("z", ascending=False)
        top = [x.upper() for x in sub.gene.head(top_n)]
        topset, N = set(top), len(top)
        res = []
        for ct, s in sets.items():
            ov = topset & s
            p = stats.hypergeom.sf(len(ov) - 1, M, len(s), N) if ov else 1.0
            res.append((ct, len(ov), len(s), p, ",".join(sorted(ov))))
        res = pd.DataFrame(res, columns=["cell_type", "overlap", "set_size", "pval", "genes"])
        res["padj"] = stats.false_discovery_control(res.pval.values, method="bh")
        res = res.sort_values(["padj", "overlap"], ascending=[True, False]).head(report)
        res.insert(0, "group", g)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)



def run_markers(cfg, X, genes, clusterings, outdir,dataset_name):
    keep = np.asarray((X > 0).sum(0)).ravel() >= cfg["min_cells"]
    Xf, genes_f = X[:, keep], genes[keep]
    print(f"\n[markers] {Xf.shape[1]} genes after min-cells filter")
    Xlog = log_normalize(Xf)

    db = None
    if cfg["marker_db"]:
        db = pd.read_csv(_path(cfg["marker_db"]))
        db = db.rename(columns={cfg["db_celltype_col"]: "cell_type", cfg["db_gene_col"]: "gene"})
        db = db[["cell_type", "gene"]].dropna()

    k = cfg["top_k"]
    summaries = []
    for name, labels in clusterings.items():
        groups = sorted_groups(labels)
        print(f"[markers] {name}: {len(groups)} clusters; running Wilcoxon tests ...")
        de = wilcoxon_one_vs_rest(Xlog, labels, groups, genes_f)
        de.to_csv(os.path.join(outdir, f"{dataset_name}_{name}_de_all.csv"), index=False)

        markers = select_markers(de, groups, k, cfg["padj"], cfg["min_log2fc"],
                                 cfg["min_pct"], cfg["unique"])
        markers.to_csv(os.path.join(outdir, f"{dataset_name}_{name}_markers_top{k}.csv"), index=False)
        missing = [g for g in groups if g not in set(markers.group)]
        if missing:
            print(f"[markers] {name}: no gene passed the filters for clusters: {missing}")

        dotplot(Xlog, labels, groups, markers, genes_f,
                os.path.join(outdir, f"{dataset_name}_{name}_dotplot.pdf"),
                title=name if cfg.get("plot_title", True) else None,
                scale_per_gene=cfg["scale_per_gene"],
                ylabel="Ground-truth label" if name.lower() in ("truth", "true") else "Predicted cluster")
        summaries.append(specificity_summary(markers, groups, k, name))

        if db is not None:
            ann = marker_db_overlap(de, groups, db, genes_f, top_n=cfg["db_top_n"])
            ann.to_csv(os.path.join(outdir, f"{dataset_name}_{name}_reference_overlap.csv"), index=False)

    summ = pd.DataFrame(summaries)
    summ.to_csv(os.path.join(outdir, f"{dataset_name}_specificity_summary.csv"), index=False)
    print(summ.to_string(index=False))


def run_dataset(ds, cfg=CONFIG):
    name = ds["name"]
    outdir = _path(os.path.join(cfg["outdir_root"], name))
    os.makedirs(outdir, exist_ok=True)
    print(f"\n############## {name} ##############")

    clusterings = {}
    for lname, spec in ds["labels"].items():
        path, col = spec.rsplit("#", 1) if "#" in spec else (spec, None)
        clusterings[lname] = load_labels(_path(path), col)
    sizes = {len(v) for v in clusterings.values()}
    if len(sizes) != 1:
        raise ValueError(f"{name}: label sources have different numbers of cells: {sizes}")
    n_cells = sizes.pop()

    names = list(clusterings)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ct = pd.crosstab(clusterings[names[i]], clusterings[names[j]]).values
            if ((ct > 0).sum(1) == 1).all() and ((ct > 0).sum(0) == 1).all():
                print(f"NOTE: '{names[i]}' and '{names[j]}' are identical partitions "
                      "(same cells grouped together, possibly with different cluster numbers).")
    if "truth" in clusterings:
        k_true = len(np.unique(clusterings["truth"]))
        for lname, lab in clusterings.items():
            if lname != "truth" and len(np.unique(lab)) < k_true:
                print(f"NOTE: '{lname}' has {len(np.unique(lab))} non-empty clusters for "
                      f"{k_true} ground-truth labels.")

    X, genes = load_expression(_path(ds["expr"]), n_cells, ds.get("x_key", cfg["x_key"]),
                               ds.get("gene_key", cfg["gene_key"]),
                               ds.get("gene_col", cfg["gene_col"]))
    if ds.get("gene_file"):
        names = np.loadtxt(_path(ds["gene_file"]), dtype=str, delimiter="\n", ndmin=1)
        if len(names) != X.shape[1]:
            raise ValueError(f"{name}: gene_file has {len(names)} names but the matrix has "
                             f"{X.shape[1]} genes.")
        genes = names
        print(f"Gene names: {ds['gene_file']}, e.g. {_values_summary(genes)}")
    if not _looks_like_symbols(genes):
        warnings.warn(f"{name}: gene names look like numbers ({_values_summary(genes)}); "
                      "the marker genes will not be interpretable.")
    print(f"Expression: {X.shape[0]} cells x {X.shape[1]} genes")

    run_markers(cfg, X, genes, clusterings, outdir,name)
    print(f"Results written to: {outdir}")


def main(datasets=DATASETS, cfg=CONFIG):
    for ds in datasets:
        run_dataset(ds, cfg)


if __name__ == "__main__":
    main()
