# scCGNet

Single-cell clustering with a variational graph autoencoder, a Gaussian mixture copula model, and a ZINB decoder, trained end to end.

## Requirements

Python 3 with:

```
torch  torch_geometric  scanpy  anndata  numpy  scipy  pandas
scikit-learn  umap-learn  matplotlib  networkx  h5py  pot
```

## Data

Put each dataset in `data/` as an `.h5ad` file named after the dataset, e.g.

```
data/Young.h5ad
data/Quake_Smart-seq2_Diaphragm.h5ad
```

Each file must contain raw counts in `adata.X` and the cell-type labels in `adata.obs["cell_type1"]` or `adata.obs["celltype"]`.

## Run

1. Choose the datasets and seeds in `main.py` (`datasetnam` and `seeds`).
2. Hyperparameters for each dataset are in `config.py`.
3. Run:

```
python main.py
```

## Output

For each dataset and seed, `main.py` writes:

- the clustering scores ( ARI, NMI) to `results_path/<dataset>.txt`;
- the predicted and true labels (`types.txt`), the latent embedding (`embedding.npy`), and UMAP plots to `results_path/plots/seed_<seed>/<dataset>/`.

