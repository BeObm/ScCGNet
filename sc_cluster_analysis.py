
import os
import warnings

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
CONFIG = {


    # HDF5 keys. Leave as None to detect them automatically.
    "x_key": None,
    "gene_key": None,
    # Clusterings to analyse: name -> "file#column" (one row per cell).

    # ---- Part 1: marker genes ----
    "run_markers": True,
    "top_k": 3,
    "padj": 0.05,
    "min_log2fc": 0.25,
    "min_pct": 0.1,        # min fraction of in-cluster cells expressing the gene
    "min_cells": 3,        # drop genes detected in fewer cells
    "unique": True,        # each marker gene shown for one cluster only
    "scale_per_gene": True,
    "plot_title": False,   # set True to print the clustering name above each dot plot
    # Optional reference markers, e.g. a CellMarker export (None to skip).
    "marker_db": None,
    "db_celltype_col": "cell_name",
    "db_gene_col": "Symbol",
    "db_top_n": 50,

    # ---- Part 2: QC comparison ----
    "qc": {
        "run": True,
        "clustering": "scCGNet",     # a name from CONFIG["labels"]
        "truth": "truth",            # a name from CONFIG["labels"] (None if absent)
        "compare": ("9", "5"),       # (cluster A, cluster B)
        "restrict_to_truth": "0",    # only use cells of this label (None = all cells)
        "mito_prefix": "MT-",        # mitochondrial gene prefix (case-insensitive)
    },
}
# =============================================================================



def _path(p):
    return p if p is None or os.path.isabs(p) else os.path.join(HERE, p)


def _decode(arr):
    return np.array([a.decode() if isinstance(a, bytes) else str(a) for a in arr])


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _describe_h5(f):
    import h5py

    lines = []
    f.visititems(lambda k, o: lines.append(
        f"  {k}: {o.shape} {o.dtype}" if isinstance(o, h5py.Dataset) else f"  {k}/ (group)"))
    return "\n".join(lines)


def _read_h5(path, x_key, gene_key, n_cells):
    import h5py

    with h5py.File(path, "r") as f:
        enc = f.attrs.get("encoding-type", b"")
        is_anndata = (enc in ("anndata", b"anndata")) or (
            "X" in f and "var" in f and "_index" in f["var"].attrs)
        if is_anndata:  # AnnData file saved with a .h5 extension
            import anndata

            ad = anndata.read_h5ad(path)
            X = ad.layers["counts"] if "counts" in ad.layers else ad.X
            return sp.csr_matrix(X), np.asarray(ad.var_names)

        dense, sparse = {}, {}

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and {"data", "indices", "indptr"} <= set(obj.keys()):
                sparse[name] = obj
            elif isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.dtype.kind in "iuf":
                dense[name] = obj

        f.visititems(visit)

        # --- count matrix ---
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
            if not cands:
                raise ValueError(
                    f"No gene-name dataset of length {X.shape[1]} found. Set CONFIG['gene_key'].\n"
                    "Contents of the .h5 file:\n" + _describe_h5(f))
            gene_key = max(cands)[1]
        genes = _decode(f[gene_key][...])
        print(f"Gene names: '{gene_key}' (e.g. {', '.join(genes[:5])})")
    return X, genes


def load_expression(path, n_cells, x_key=None, gene_key=None):
    """Return (X, genes) with X of shape (cells, genes), raw counts for all genes."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        X, genes = _read_h5(path, x_key, gene_key, n_cells)
    elif ext == ".h5ad":
        import anndata

        ad = anndata.read_h5ad(path)
        X = ad.layers["counts"] if "counts" in ad.layers else ad.X
        genes = np.asarray(ad.var_names)
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


def load_labels(path, column=None):
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


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Differential expression
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Dot plot
# ----------------------------------------------------------------------------
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



METRICS = {
    "library_size": "Library size (total counts)",
    "detected_genes": "Detected genes",
    "mito_fraction": "Mitochondrial fraction",
}


def compute_qc(X, genes, prefix):
    """Per-cell QC measures from a raw count matrix (cells x genes, sparse)."""
    lib = np.asarray(X.sum(1)).ravel()
    det = np.asarray((X > 0).sum(1)).ravel()
    mito = np.array([g.upper().startswith(prefix.upper()) for g in genes])
    qc = pd.DataFrame({"library_size": lib, "detected_genes": det})
    if mito.any():
        mito_counts = np.asarray(X[:, mito].sum(1)).ravel()
        qc["mito_fraction"] = np.divide(mito_counts, lib, out=np.zeros_like(lib, dtype=float),
                                        where=lib > 0)
        print(f"Mitochondrial genes found: {mito.sum()} (e.g. {', '.join(genes[mito][:5])})")
    else:
        print(f"WARNING: no gene name starts with '{prefix}'. The mitochondrial fraction "
              "cannot be computed (mitochondrial genes were probably removed in preprocessing).")
    return qc


def summarize(values):
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return med, q1, q3


def compare(qc, a, b):
    rows = []
    for m in [c for c in METRICS if c in qc]:
        xa = qc.loc[qc.cluster == a, m].values
        xb = qc.loc[qc.cluster == b, m].values
        res = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        med_a, q1_a, q3_a = summarize(xa)
        med_b, q1_b, q3_b = summarize(xb)
        rows.append({
            "metric": m,
            f"n_{a}": len(xa), f"median_{a}": med_a, f"IQR_{a}": f"{q1_a:.4g}-{q3_a:.4g}",
            f"n_{b}": len(xb), f"median_{b}": med_b, f"IQR_{b}": f"{q1_b:.4g}-{q3_b:.4g}",
            # AUC = P(value in A > value in B) + 0.5 P(tie); 0.5 means no difference
            "AUC_A_gt_B": res.statistic / (len(xa) * len(xb)),
            "pval_reference_only": res.pvalue,
        })
    return pd.DataFrame(rows)


def boxplots(qc, a, b, out_path, title):
    metrics = [c for c in METRICS if c in qc]
    groups = sorted_groups(qc.cluster.values)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        data = [qc.loc[qc.cluster == g, m].values for g in groups]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor("#d6604d" if g == a else "#4393c3" if g == b else "white")
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_xlabel("Predicted cluster")
        ax.set_ylabel(METRICS[m])
        if m == "library_size":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Part 1: marker genes
# ----------------------------------------------------------------------------
def run_markers(cfg, X, genes, clusterings, outdir):
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
        de.to_csv(os.path.join(outdir, f"{name}_de_all.csv"), index=False)

        markers = select_markers(de, groups, k, cfg["padj"], cfg["min_log2fc"],
                                 cfg["min_pct"], cfg["unique"])
        markers.to_csv(os.path.join(outdir, f"{name}_markers_top{k}.csv"), index=False)
        missing = [g for g in groups if g not in set(markers.group)]
        if missing:
            print(f"[markers] {name}: no gene passed the filters for clusters: {missing}")

        dotplot(Xlog, labels, groups, markers, genes_f,
                os.path.join(outdir, f"{name}_dotplot.pdf"),
                title=name if cfg.get("plot_title", True) else None,
                scale_per_gene=cfg["scale_per_gene"],
                ylabel="Ground-truth label" if name.lower() in ("truth", "true") else "Predicted cluster")
        summaries.append(specificity_summary(markers, groups, k, name))

        if db is not None:
            ann = marker_db_overlap(de, groups, db, genes_f, top_n=cfg["db_top_n"])
            ann.to_csv(os.path.join(outdir, f"{name}_reference_overlap.csv"), index=False)

    summ = pd.DataFrame(summaries)
    summ.to_csv(os.path.join(outdir, "specificity_summary.csv"), index=False)
    print(summ.to_string(index=False))


# ----------------------------------------------------------------------------
# Part 2: QC comparison
# ----------------------------------------------------------------------------
def run_qc(qcfg, X, genes, clusterings, outdir):
    print("\n[QC] computed from all genes in the expression file. If it contains only "
          "selected genes (e.g. highly variable genes), library size and detected genes are partial.")
    if qcfg["clustering"] not in clusterings:
        raise ValueError(f"QC clustering '{qcfg['clustering']}' is not in CONFIG['labels'].")
    clusters = clusterings[qcfg["clustering"]]
    truth = clusterings.get(qcfg["truth"]) if qcfg.get("truth") else None

    qc = compute_qc(X, genes, qcfg["mito_prefix"])
    qc.insert(0, "cluster", clusters)
    if truth is not None:
        qc.insert(1, "truth", truth)
    qc.to_csv(os.path.join(outdir, "qc_per_cell.csv"), index_label="cell")

    rows = []
    for g in sorted_groups(clusters):
        sub = qc[qc.cluster == g]
        row = {"cluster": g, "n_cells": len(sub)}
        for m in [c for c in METRICS if c in qc]:
            med, q1, q3 = summarize(sub[m].values)
            row.update({f"{m}_median": med, f"{m}_q1": q1, f"{m}_q3": q3})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "qc_by_cluster.csv"), index=False)

    a, b = (str(c) for c in qcfg["compare"])
    sub, note = qc, "all cells"
    if qcfg.get("restrict_to_truth") is not None and truth is not None:
        sub = qc[qc.truth == str(qcfg["restrict_to_truth"])]
        note = f"cells of ground-truth label {qcfg['restrict_to_truth']} only"
    for c in (a, b):
        if (sub.cluster == c).sum() < 2:
            raise ValueError(f"Cluster {c} has fewer than 2 cells ({note}).")
    comp = compare(sub, a, b)
    comp.insert(0, "cells_used", note)
    comp.to_csv(os.path.join(outdir, "qc_comparison.csv"), index=False)

    boxplots(qc, a, b, os.path.join(outdir, "qc_boxplots.pdf"),
             f"QC per predicted cluster (cluster {a} red, cluster {b} blue; all cells)")
    if note != "all cells":
        boxplots(sub[sub.cluster.isin([a, b])], a, b,
                 os.path.join(outdir, "qc_boxplots_restricted.pdf"),
                 f"Clusters {a} and {b}, {note}")

    pd.set_option("display.width", 200)
    print(f"[QC] cluster {a} vs {b} ({note}):")
    print(comp.drop(columns="cells_used").to_string(index=False))
    print("AUC = probability that a cell of cluster A has a higher value than a cell of "
          "cluster B (0.5 = no difference).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(cfg=CONFIG):
    outdir = _path(cfg["outdir"])
    os.makedirs(outdir, exist_ok=True)

    # labels first: their length fixes the number of cells
    clusterings = {}
    for name, spec in cfg["labels"].items():
        path, col = spec.rsplit("#", 1) if "#" in spec else (spec, None)
        clusterings[name] = load_labels(_path(path), col)
    n_cells = {len(v) for v in clusterings.values()}
    if len(n_cells) != 1:
        raise ValueError(f"Label files have different numbers of cells: {n_cells}")
    n_cells = n_cells.pop()

    X, genes = load_expression(_path(cfg["expr"]), n_cells, cfg["x_key"], cfg["gene_key"])
    print(f"Expression: {X.shape[0]} cells x {X.shape[1]} genes")
    print("NOTE: the cell order of the matrix must match the labels file; this cannot be checked here.")

    if cfg.get("run_markers", True):
        run_markers(cfg, X, genes, clusterings, outdir)
    if cfg.get("qc", {}).get("run", False):
        run_qc(cfg["qc"], X, genes, clusterings, outdir)
    print(f"\nResults written to: {outdir}")


if __name__ == "__main__":
    label1= [{
        "truth": "data/young_types.txt#True",
        "scCGNet": "data/young_types.txt#Pred",
        "GMM": "data/GMM_data/young_types.txt#Pred",
    },
    "sc_cluster_analysis_Young", "data/young.h5ad",
    {
        "run": True,
        "clustering": "scCGNet",  # a name from CONFIG["labels"]
        "truth": "truth",  # a name from CONFIG["labels"] (None if absent)
        "compare": ("9", "5"),  # (cluster A, cluster B)
        "restrict_to_truth": "0",  # only use cells of this label (None = all cells)
        "mito_prefix": "MT-",  # mitochondrial gene prefix (case-insensitive)
    },
    ]

    labels2= [{
        "truth": "data/diaphragm_types.txt#True",
        "scCGNet": "data/diaphragm_types.txt#Pred",
        "GMM": "data/GMM_data/diaphragm_types.txt#Pred",
    },
    "sc_cluster_analysis_Diaphragm", "data/Quake_Smart-seq2_Diaphragm.h5ad",
        {
        "run": False,
        "clustering": "GMM",  # a name from CONFIG["labels"]
        "truth": "truth",  # a name from CONFIG["labels"] (None if absent)
        "compare": ("9", "5"),  # (cluster A, cluster B)
        "restrict_to_truth": "0",  # only use cells of this label (None = all cells)
        "mito_prefix": "MT-",  # mitochondrial gene prefix (case-insensitive)
    },
    ]


    for i,label in enumerate([label1,labels2]):
        print("############## Dataset: ",i+1, "##############")
        cfg = CONFIG.copy()
        cfg["labels"] = label[0]
        cfg["outdir"] = label[1]
        cfg["expr"] = label[2]
        cfg["qc"] = label[3]

        main(cfg)