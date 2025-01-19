import scanpy as sc
import numpy as np
import pandas as pd
import random
class DiffusionDataset:
    def __init__(self):
        self.adata = DiffusionDataset.getSingleCellSource()
        self.bulkRNA = DiffusionDataset.getBulkRNASource()

    @staticmethod
    def getSingleCellData():
        adata = sc.read_mtx("data/GSM4041646_3_cell-line_mixture.cell_ranger.matrix.mtx")
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
    def getBulkRNAData():
        data = pd.read_excel("data/GSE136148_Bulk_rawgenetable.xlsx", sheet_name= "3 Cell-line Bulk", index_col= 0)
        return data
    def getBulkRNASource():
        data = DiffusionDataset.getBulkRNAData()
        total = data.sum().values[0]
        dataNormalized = (data / total) * 1e4
        dataLog1p = np.log1p(dataNormalized)
        return dataLog1p
if __name__ == "__main__":
    data = DiffusionDataset()
    print(f'scRNA-seq:{data.adata}')
    print(f'bulkRNA:{data.bulkRNA}')
    