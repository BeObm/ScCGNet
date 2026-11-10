from preprocessing import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse
from vgae_model import GraphVAE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from config import configs
import umap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc, anndata as ad, pandas as pd
from common_code.get_labels import *



@torch.no_grad()
def kmeans_warmstart(model, x_input, edge_index, n_clusters, seed=0):

    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)  # deterministic
    z = mu.detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(z)
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=mu.device)
    model.cluster.mu.data.copy_(centers)
    return km.labels_


@torch.no_grad()
def evaluate(model, x_input, edge_index, labels):
    """ARI / NMI of GMCM hard assignments vs ground truth (labels: eval only)."""
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)
    pred = model.cluster.assign(mu).cpu().numpy()
    acc=compute_cluster_accuracy(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return ari, nmi,acc


def train(model, x_input, edge_index, adj, x_counts, labels,
          scale_factor=1.0, n_clusters=None,
          pretrain_epochs=200, train_epochs=300, lr=1e-3,
          weights=(1.0, 1.0, 1.0, 1.0), eval_every=10, device="cpu"):

    w_adj, w_feat, w_clus, w_kl = weights
    model = model.to(device)
    x_input, edge_index = x_input.to(device), edge_index.to(device)
    adj, x_counts = adj.to(device), x_counts.to(device)
    if torch.is_tensor(scale_factor):
        scale_factor = scale_factor.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Phase 1: reconstruction pretraining (clustering weight = 0) ----
    for ep in range(pretrain_epochs):
        model.train()
        opt.zero_grad()
        loss, parts = model.loss(x_input, edge_index, adj, x_counts, scale_factor,
                                 w_adj=w_adj, w_feat=w_feat, w_clus=0.0, w_kl=w_kl)
        loss.backward()
        opt.step()
        if ep % eval_every == 0:
            print(f"[pretrain {ep:4d}] total={parts['total']:.4f} "
                  f"adj={parts['adj']:.4f} feat={parts['feat']:.4f} kl={parts['kl']:.4f}")

    kmeans_warmstart(model, x_input, edge_index, n_clusters)
    ari, nmi,acc = evaluate(model, x_input, edge_index, labels)
    print(f"[k-means init] ARI={ari:.4f} NMI={nmi:.4f}")

    best = {"acc":-1,"ari": -1.0, "nmi": -1.0, "epoch": -1}
    for ep in range(train_epochs):
        model.train()

        opt.zero_grad()
        loss, parts = model.loss(x_input, edge_index, adj, x_counts, scale_factor,
                                 w_adj=w_adj, w_feat=w_feat, w_clus=w_clus, w_kl=w_kl)
        loss.backward()
        opt.step()
        if ep % eval_every == 0:
            ari, nmi,acc = evaluate(model, x_input, edge_index, labels)
            if ari > best["ari"]:
                best = {"acc":acc,"ari": ari, "nmi": nmi, "epoch": ep}
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[train {ep:4d}] | ARI={ari:.4f} NMI={nmi:.4f}")
    model.load_state_dict(best_state)
    print(f"[best] epoch={best['epoch']} ARI={best['ari']:.4f} NMI={best['nmi']:.4f}")
    return model, best


@torch.no_grad()
def save_plots(model, x_input, edge_index, labels, out_prefix,seed,dataset="Adam"):
    os.makedirs(out_prefix, exist_ok=True)
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)
    z = mu.cpu().numpy()
    pred = model.cluster.assign(mu).cpu().numpy()

    cell_ids = np.arange(len(labels))
    newdata_path = f"cluster_data/seed_{seed}/{dataset}"
    os.makedirs(newdata_path, exist_ok=True)
    run(cell_ids=cell_ids, y_true=labels, y_pred=pred, dataset_name=dataset, model_name="scCGNet",
        out_path=f"{newdata_path}/scCGNet.csv",seed=seed)


    u = umap.UMAP(random_state=0).fit_transform(z)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, (c, t) in zip(ax, [(labels, "True"), (pred, "Pred")]):
        sc = a.scatter(u[:, 0], u[:, 1], c=c, cmap="tab20", s=6)
        a.set_title(t); a.set_xlabel("UMAP-1"); a.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=a)
    fig.tight_layout(); fig.savefig(f"{out_prefix}/umap.png", dpi=200); plt.close(fig)

    np.save(f"{out_prefix}/embedding.npy", z)
    import pandas as pd
    pd.DataFrame({"True": labels, "Pred": pred}).to_csv(f"{out_prefix}/types.txt")
    print("saved", out_prefix)


@torch.no_grad()
def save_trajectory(model, x_input, edge_index, out_prefix, root_cluster=0):
    os.makedirs(out_prefix, exist_ok=True)
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)
    z = mu.cpu().numpy()
    pred = model.cluster.assign(mu).cpu().numpy()

    a = ad.AnnData(z)
    a.obs["cluster"] = pd.Categorical(pred.astype(str))
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15)
    sc.tl.paga(a, groups="cluster")
    sc.pl.paga(a, show=False)
    plt.savefig(f"{out_prefix}/paga.png", dpi=200); plt.close()

    sc.tl.umap(a, init_pos="paga")
    a.uns["iroot"] = int(np.where(pred == root_cluster)[0][0])
    sc.tl.diffmap(a); sc.tl.dpt(a)

    sc.pl.umap(a, color=["cluster", "dpt_pseudotime"], show=False)
    plt.savefig(f"{out_prefix}/trajectory.png", dpi=200); plt.close()
    print("saved trajectory", out_prefix)



    sc.pp.neighbors(a, use_rep="X", n_neighbors=15)
    sc.tl.paga(a, groups="cluster")
    sc.tl.umap(a, init_pos="paga")
    a.uns["iroot"] = int(np.where(pred == root_cluster)[0][0])
    sc.tl.diffmap(a); sc.tl.dpt(a)

    sc.pl.paga(a, show=False)
    plt.savefig(f"{out_prefix}/paga.png", dpi=200); plt.close()
    sc.pl.umap(a, color=["cluster", "dpt_pseudotime"], show=False)
    plt.savefig(f"{out_prefix}/trajectory.png", dpi=200); plt.close()
    print("saved trajectory", out_prefix)


def load_my_data(dataPath, dataset, hvg, n_neighbors=15, ts=None, metric='cosine'):
    adata = ad.read_h5ad(dataPath)

    if "n_counts" not in adata.obs.columns:
        adata.obs['n_counts'] = np.asarray(adata.X.sum(axis=1)).ravel()

    print("obs columns:", adata.obs.columns.tolist())


    CELLTYPE_KEYS = [
        "cell_type1", "celltype", "cell_type", "CellType",
        "cell_ontology_class", "free_annotation", "louvain", "labels", "label",
    ]
    ct_col = next((k for k in CELLTYPE_KEYS if k in adata.obs.columns), None)
    if ct_col is None:
        raise KeyError(
            f"No cell-type column found. Looked for {CELLTYPE_KEYS}; "
            f"available obs columns: {adata.obs.columns.tolist()}"
        )
    print(f"{dataset}: using cell-type column '{ct_col}'")

    n_before = adata.n_obs
    keep = adata.obs[ct_col].notna().to_numpy()
    if not keep.all():
        adata = adata[keep].copy()
        print(f"{dataset}: dropped {n_before - adata.n_obs} unannotated cells "
              f"({adata.n_obs} remain)")

    # --- standard preprocessing ---
    scanpy.pp.filter_cells(adata, min_genes=1)
    scanpy.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata
    adata.X = adata.X.astype(np.float32)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    scanpy.pp.normalize_total(adata)
    scanpy.pp.log1p(adata)
    scanpy.pp.highly_variable_genes(adata, n_top_genes=hvg)
    adata.raw.var['highly_variable'] = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]

    rawData = adata.raw[:, adata.raw.var['highly_variable']].X

    # --- cell-cell graph ---
    adj, r_adj = adata_knn(adata, method='gauss', knn=True,
                           n_neighbors=n_neighbors, metric=metric)
    adj = adj.toarray()
    adj[adj > 0] = int(1)
    edge_index = np.where(adj > 0)
    edge_index = np.concatenate((np.expand_dims(edge_index[0], axis=0),
                                 np.expand_dims(edge_index[1], axis=0)), axis=0)

    # --- gene-gene graph ---
    adj_gene = kneighbors_graph(rawData.T, n_neighbors=5,
                                mode='connectivity', include_self=True)
    if issparse(adj_gene):
        adj_gene = adj_gene.toarray()
    edge_index_g = np.where(adj_gene > 0)
    edge_index_g = np.concatenate((np.expand_dims(edge_index_g[0], axis=0),
                                   np.expand_dims(edge_index_g[1], axis=0)), axis=0)

    feature = rawData
    if issparse(feature):
        feature = feature.toarray()

    # --- labels: integer codes + readable names, uniform across all datasets ---
    celltype = adata.obs[ct_col].astype("category")
    label = celltype.cat.codes.to_numpy()
    label_names = celltype.cat.categories.to_list()
    cell_type_names = celltype.astype(str).to_numpy()

    features = get_feat_input(feature, adj, ts)
    n_classes = len(label_names)

    print(f"{dataset}: {n_classes} cell types -> {label_names}")

    return {
        "features": features,
        "label": label,
        "label_names": label_names,
        "cell_type_names": cell_type_names,
        "n_classes": n_classes,
        "edge_index": edge_index,
        "edge_index_g": edge_index_g,
        "celltype": celltype,
    }



if __name__ == "__main__":
    os.makedirs("Result_GMM/results", exist_ok=True)
    parser = argparse.ArgumentParser(
        description=" scRNA-seq clustering with GMCM-VGAE")

    parser.add_argument("--dataset_name", type=str, default="Quake_10x_Limb_Muscle")
    args = parser.parse_args()
    device = get_device()
    datasetnam = [
        "Adam",
        "Muraro",
        "Quake_10x_Bladder",
        "Quake_10x_Limb_Muscle",
        "Quake_10x_Spleen",
        "Quake_Smart-seq2_Diaphragm",
        "Quake_Smart-seq2_Limb_Muscle",
        "Quake_Smart-seq2_Lung",
        "Quake_Smart-seq2_Trachea",
        "Romanov",
        "Young"
        "Tosches_turtle",
        "Wang_Large_Intestine"
        "Campbell",
        "Cao_2020_Spleen",
        "Bach",
        "Shekhar"
        ]

    # seeds = [111,222,333, 444, 555]
    seeds = [333]

    for dataset in datasetnam:
            args.dataset_name=dataset

            n_eighborss = 15
            n_top_genes = 2000
            datapath = f"./data/{args.dataset_name}.h5ad"
            label = load_my_data(dataPath=datapath,
                                       dataset=args.dataset_name,
                                       hvg=n_top_genes,
                                       n_neighbors=n_eighborss,
                                       ts=[0, 0],
                                       metric='cosine')
            print(dataset,label)