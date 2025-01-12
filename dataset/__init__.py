import scanpy as sc
import numpy as np
import random
class DiffusionDataset:
    def __init__(self):
        self.adata = DiffusionDataset.getSingleCellSource()
        self.bulkRNA = DiffusionDataset.getBulkRNAData(self)

    @staticmethod
    def getSingleCellData():
        adata = sc.datasets.pbmc3k()
        return adata
    
    @staticmethod
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
    def getBulkRNAData(self, numBulk=100, cellsPerBulk=20):
        adata = self.adata
        bulkRNADict = {}
        uniqueCellType = adata.obs["leiden"].unique()
        cellTypeProportion = {cellType: random.uniform(0.05,0.2) for cellType in uniqueCellType}
        totalProportion = sum(cellTypeProportion.values())
        cellTypeProportion = {k: v/totalProportion for k, v in cellTypeProportion.items()}
        # print("細胞類型及其細胞數量:")
        # for cellType in uniqueCellType:
        #     num_cells = adata.obs[adata.obs["leiden"] == cellType].shape[0]
        #     print(f"Cell type {cellType}: {num_cells} cells")
        for bulk_idx in range(numBulk):
            selectedCells = []
            for cellType, proportion in cellTypeProportion.items():
                numCells = int(cellsPerBulk * proportion)
                subset = adata[adata.obs["leiden"] == cellType]
                replace = subset.n_obs < numCells
                selectedIndices = np.random.choice(subset.n_obs, numCells, replace=replace)
                # print("selecI:",selectedIndices)
                selectedCells.append(subset.X[selectedIndices,:].toarray())
            bulkExpression = np.vstack(selectedCells).mean(axis=0)
            # print("expression",bulkExpression)
            bulkRNADict[f"Bulk_{bulk_idx}"] = np.asarray(bulkExpression).flatten()
        # print("bulkRNADict",bulkRNADict)
        return bulkRNADict
if __name__ == "__main__":
    data = DiffusionDataset()
    adata = data.adata
    bulkRNA = data.bulkRNA
    # print(bulkRNA)
    # print(adata)
    # print(bulkRNA.shape)
    # print(adata.obs["leiden"])