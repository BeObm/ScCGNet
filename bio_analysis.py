import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DATA_DIR = "data"                 # folder holding the .h5ad and types.txt files
OUT_DIR  = "bio_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


DATASETS = {
    "Diaphragm": ("Quake_Smart-seq2_Diaphragm.h5ad", "diaphragm_types.txt", 2000),
    "Young":     ("Young.h5ad",                       "young_types.txt",     2000),
    "Bladder":   ("Quake_10x_Bladder.h5ad",           "bladder_types.txt",   2000),
    "Qx_Muscle": ("Quake_10x_Limb_Muscle.h5ad",       "qx_muscle_types.txt",    2000),
    "QS_Muscle": ("Quake_Smart-seq2_Limb_Muscle.h5ad",       "qs_muscle_types.txt",    2000),
    "Lung": ("Quake_Smart-seq2_Lung.h5ad",       "lung_types.txt",    2000),

}

CELLTYPE_KEYS = ["cluster","cell_type1", "celltype", "cell_type", "CellType",
                 "cell_ontology_class", "free_annotation"]


def _is_string_names(categories):
    """True only if the categories are genuine text labels, not numbers-as-strings."""
    for c in categories:
        if isinstance(c, (int, np.integer, float, np.floating)):
            return False
        if isinstance(c, str):
            s = c.strip()
            if s.lstrip("-").replace(".", "", 1).isdigit():
                return False
    return len(categories) > 0


def get_celltype_names(h5_path, hvg,
                       celltype_keys=None,
                       label_candidates=("cell_type1", "celltype")):
    """Recover per-cell label CODES (from the same column training used) and,
    when available, readable NAMES.

    label_candidates : columns to use as the CODE source, in the SAME priority
        order as load_h5_data (cell_type1 first, else celltype). The first one
        present is used, so codes stay aligned with the saved types.txt.
    celltype_keys : columns to search for real string NAMES.

    Returns: codes, code_names (index=code -> name), has_names, name_col
    """
    if celltype_keys is None:
        celltype_keys = [
            "cell_ontology_class", "free_annotation",
            "celltype", "cell_type", "CellType", "cell_type1",
        ]

    adata = ad.read_h5ad(h5_path)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)

    label_col = next((c for c in label_candidates if c in adata.obs.columns), None)
    if label_col is None:
        raise KeyError(
            f"{h5_path}: none of the label candidates {label_candidates} found; "
            f"have {adata.obs.columns.tolist()}"
        )
    label_cat = pd.Categorical(adata.obs[label_col])
    codes = np.asarray(label_cat.codes)
    n_codes = len(label_cat.categories)

    name_col = None
    for key in celltype_keys:
        if key not in adata.obs.columns:
            continue
        cats = list(pd.Categorical(adata.obs[key]).categories)
        if _is_string_names(cats):
            name_col = key
            break
        else:
            print(f"  [{os.path.basename(h5_path)}] '{key}' is numeric -- skipping.")

    if name_col is None:
        print(f"  [{os.path.basename(h5_path)}] no name column; using codes "
              f"(label col '{label_col}').")
        return codes, [f"type {i}" for i in range(n_codes)], False, None

    if name_col == label_col:
        code_names = [str(c) for c in label_cat.categories]
        return codes, code_names, True, name_col

    tmp = pd.DataFrame({"code": codes,
                        "name": adata.obs[name_col].astype(str).values})
    code2name = tmp.groupby("code")["name"].agg(lambda s: s.mode().iloc[0])
    purity = tmp.groupby("code")["name"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0])
    impure = purity[purity < 0.99]
    if len(impure):
        print(f"  [{os.path.basename(h5_path)}] WARNING: label '{label_col}' and "
              f"name '{name_col}' disagree for codes {list(impure.index)} "
              f"(purity {impure.round(2).to_dict()}); names are majority vote.")
    code_names = [code2name.get(i, f"type {i}") for i in range(n_codes)]
    return codes, code_names, True, name_col


def analyse(true, pred, code_names):
    counts = pd.crosstab(pd.Series(true, name="cell type"),
                         pd.Series(pred, name="cluster"))
    frac = pd.crosstab(pd.Series(true, name="cell type"),
                       pd.Series(pred, name="cluster"), normalize="index")

    per_type = pd.DataFrame({
        "cell_type": [code_names[c] if c < len(code_names) else f"type {c}"
                      for c in counts.index],
        "n_cells":   counts.sum(axis=1).values,
        "maps_to":   frac.idxmax(axis=1).values,
        "purity":    frac.max(axis=1).round(3).values,
    })
    frac.index = [code_names[c] if c < len(code_names) else f"type {c}"
                  for c in frac.index]

    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    return counts, frac, per_type, ari, nmi


def plot_heatmap(frac, name, ari, nmi, out_png):
    fig, ax = plt.subplots(figsize=(1.1 * frac.shape[1] + 3.5,
                                    0.55 * frac.shape[0] + 2.0))
    im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(frac.shape[1])); ax.set_xticklabels(frac.columns)
    ax.set_yticks(range(frac.shape[0])); ax.set_yticklabels(frac.index, fontsize=8)
    ax.set_xlabel("predicted cluster"); ax.set_ylabel("annotated cell type")
    ax.set_title(f"{name}   (ARI={ari:.3f}, NMI={nmi:.3f})", fontsize=10)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = []
    for name, (h5, txt, hvg) in DATASETS.items():
        h5_path  = os.path.join(DATA_DIR, h5)
        txt_path = os.path.join(DATA_DIR, txt)
        if not (os.path.exists(h5_path) and os.path.exists(txt_path)):
            print(f"[{name}] SKIP (missing {h5 if not os.path.exists(h5_path) else txt})")
            continue

        codes, code_names, _, col = get_celltype_names(h5_path, hvg)
        df = pd.read_csv(txt_path, index_col=0)
        true, pred = df["True"].to_numpy(), df["Pred"].to_numpy()

        # --- alignment sanity check: types.txt 'True' must equal the h5ad codes ---
        aligned = (len(true) == len(codes)) and np.array_equal(true, codes)
        if not aligned:
            print(f"[{name}] WARNING: types.txt 'True' does not match the h5ad "
                  f"cell-type codes (len {len(true)} vs {len(codes)}). "
                  f"Names may be MISALIGNED -- verify before trusting the biology.")

        counts, frac, per_type, ari, nmi = analyse(true, pred, code_names)

        counts.to_csv(os.path.join(OUT_DIR, f"{name}_confusion_counts.csv"))
        frac.round(3).to_csv(os.path.join(OUT_DIR, f"{name}_confusion_fraction.csv"))
        per_type.to_csv(os.path.join(OUT_DIR, f"{name}_recovery.csv"), index=False)
        plot_heatmap(frac, name, ari, nmi,
                     os.path.join(OUT_DIR, f"{name}_confusion.png"))

        n_clean = int((per_type["purity"] >= 0.90).sum())
        print(f"\n===== {name}  (col='{col}', n={len(true)}, "
              f"{per_type.shape[0]} types, ARI={ari:.4f}, NMI={nmi:.4f}) =====")
        print(per_type.to_string(index=False))
        print(f"cell types recovered at >=90% purity: {n_clean}/{per_type.shape[0]}")

        summary.append({"dataset": name, "cell_type_col": col, "n_cells": len(true),
                        "n_types": per_type.shape[0],
                        "recovered_>=90%": n_clean,
                        "ARI": round(ari, 4), "NMI": round(nmi, 4),
                        "aligned": aligned})

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
        print(f"\nAll outputs -> {OUT_DIR}/")
    else:
        print("No datasets processed.")


if __name__ == "__main__":
    main()