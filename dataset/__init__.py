import scanpy as sc
import numpy as np
import pandas as pd
import random
class DiffusionDataset:
    def __init__(self):
        self.adata = DiffusionDataset.getSingleCellSource()
        self.bulkRNA, self.pair = DiffusionDataset.getBulkRNAData(self)

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
    def getBulkRNAData(self, numBulk=100, cellsPerBulk=50):
        """
        生成 bulk RNA 與對應的單細胞 pair data，並保存為 CSV 文件。
        """
        adata = self.adata
        bulkRNADict = {}
        bulkRNAPairs = {}
        unique_cell_types = adata.obs["leiden"].unique()

        # 隨機生成不同細胞類型的比例，總和為 1
        cell_type_proportions = {cell_type: random.uniform(0.05, 0.2) for cell_type in unique_cell_types}
        total_proportion = sum(cell_type_proportions.values())
        cell_type_proportions = {k: v / total_proportion for k, v in cell_type_proportions.items()}

        # 生成 numBulk 個 bulk RNA 样本
        for bulk_idx in range(numBulk):
            selected_cells = []
            selected_indices = []
            for cell_type, proportion in cell_type_proportions.items():
                # 按比例選擇每個細胞類型中的細胞數量
                num_cells = int(cellsPerBulk * proportion)
                subset = adata[adata.obs["leiden"] == cell_type]
                indices = np.random.choice(subset.n_obs, num_cells, replace=False)
                selected_indices.extend(subset.obs_names[indices].tolist())
                selected_cells.append(subset.X[indices, :].toarray())

            # 將選中的細胞進行加權平均，生成 bulk RNA 樣本
            bulk_expression = np.vstack(selected_cells).mean(axis=0)
            bulk_id = f"Bulk_{bulk_idx}"
            bulkRNADict[bulk_id] = np.asarray(bulk_expression).flatten()
            bulkRNAPairs[bulk_id] = selected_indices

        # 保存 bulk RNA 数据为 CSV 文件
        bulkRNA_df = pd.DataFrame.from_dict(bulkRNADict, orient='index', columns=adata.var_names)
        bulkRNA_df.to_csv("bulk_rna_data.csv", index=True, index_label="SampleID")

        # 保存 pair data 为 CSV 文件
        pair_data = []
        for bulk_id, cell_ids in bulkRNAPairs.items():
            pair_data.append({"BulkID": bulk_id, "SingleCellIDs": ",".join(cell_ids)})
        pair_df = pd.DataFrame(pair_data)
        pair_df.to_csv("bulk_rna_pairs.csv", index=False)

        return bulkRNA_df, pair_df
if __name__ == "__main__":
    data = DiffusionDataset()
    adata = data.adata
    bulkRNA = data.bulkRNA
    pair = data.pair
    print(pair)
    # print(bulkRNA)
    # print(adata)
    # print(bulkRNA.shape)
    # print(adata.obs["leiden"])