import scanpy as sc
import numpy as np

class DiffusionDataset:
    def __init__(self):
        self.adata = DiffusionDataset.getSingleCellSource()
        self.bulkRNA = DiffusionDataset.getBulkRNAData(self)
    def getSingleCellData():
        adata = sc.datasets.pbmc3k()
        return adata
    def getSingleCellSource():
        adata = DiffusionDataset.getSingleCellData()
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15, metric='cosine')
        sc.tl.leiden(adata, resolution=0.8)
        return adata
    def getBulkRNAData(self):
        adata = self.adata
        bulkRNADict = {}
        for cellType in adata.obs["leiden"]:
            subset = adata[adata.obs["leiden"] == cellType]    
            bulkRNA = subset.X.sum(axis=0)
            bulkRNADict[cellType] = np.asarray(bulkRNA).flatten()
            # print(f"Cell type {cellType}: bulk RNA shape {bulkRNADict[cellType].shape}")
        return bulkRNADict
    

if __name__ == "__main__":
    data = DiffusionDataset()
    adata = data.adata
    bulkRNA = data.bulkRNA
    print(bulkRNA)
    # print(adata)
    # print(bulkRNA.shape)
    # print(adata.obs["leiden"])