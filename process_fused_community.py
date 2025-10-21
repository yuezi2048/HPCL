import torch
import networkx as nx
from community import community_louvain
import os
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 导入您的 Dataset 类 ---
from utils.dataset import RecDataset

# --- 配置 ---
config = {
    'dataset': 'baby',
    'data_path': '/home/ljy/Documents/nwu/science/code/TT2/data/',
    'USER_ID_FIELD': 'userID',
    'ITEM_ID_FIELD': 'itemID',
    'inter_splitting_label': 'x_label',
    'inter_file_name': 'baby.inter',
    'field_separator': '\t',
    'filter_out_cod_start_users': False
}

dataset_name = config['dataset']
dataset_path = os.path.join(config['data_path'], dataset_name)
content_knn_k = 10 # 内容图的K值
beta = 0.8 # <--- 建议先提高beta值，突出内容图的作用

# =================================================================================
# ====== 新增超参数 START =========================================================
# =================================================================================
# 为协同相似性图设置一个独立的K值
collab_knn_k = 10  # 您可以调整这个值, e.g., 10, 20, 50
# =================================================================================
# ====== 新增超参数 END ===========================================================

# 路径
output_file = os.path.join(dataset_path, 'item_to_fused_community.pt')
mm_adj_path = os.path.join(dataset_path, f'mm_adj_{content_knn_k}.pt')

# --- 1. 加载和构建交互矩阵 R (逻辑不变) ---
print("正在使用 RecDataset 加载数据集...")
dataset = RecDataset(config)
n_users = dataset.user_num
n_items = dataset.item_num
print(f"数据集信息: {n_users} 个用户, {n_items} 个物品。")
rows = dataset.df[config['USER_ID_FIELD']].values
cols = dataset.df[config['ITEM_ID_FIELD']].values
data = np.ones(len(rows), dtype=np.float32)
interaction_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

# --- 2. 计算协同相似性图 A_collab (逻辑不变) ---
print("计算物品间的协同余弦相似度...")
A_collab_dense = cosine_similarity(interaction_matrix.transpose(), dense_output=False).tocsr()
A_collab_dense.setdiag(0)
print("协同相似性图 A_collab 构建完成。")

# =================================================================================
# ====== 主要修改部分 START =======================================================
# =================================================================================

# --- 3. 对协同相似性图 A_collab 进行 Top-K 稀疏化 ---
print(f"正在对协同图进行 Top-K 稀疏化, K={collab_knn_k} ...")
# A_collab_dense 是 csr_matrix, (n_items, n_items)
n_items, _ = A_collab_dense.shape
new_rows, new_cols, new_data = [], [], []

# 遍历每一行（每个物品）
for i in range(n_items):
    # 获取当前行的非零元素
    start_ptr = A_collab_dense.indptr[i]
    end_ptr = A_collab_dense.indptr[i+1]
    if start_ptr == end_ptr:
        continue # 没有邻居

    row_indices = A_collab_dense.indices[start_ptr:end_ptr]
    row_data = A_collab_dense.data[start_ptr:end_ptr]

    # 找到Top-K的邻居
    # 如果邻居数少于K，则全部保留
    if len(row_data) > collab_knn_k:
        top_k_indices_local = np.argpartition(-row_data, collab_knn_k)[:collab_knn_k]
        top_k_indices_global = row_indices[top_k_indices_local]
        top_k_data = row_data[top_k_indices_local]
    else:
        top_k_indices_global = row_indices
        top_k_data = row_data

    new_rows.extend([i] * len(top_k_indices_global))
    new_cols.extend(top_k_indices_global)
    new_data.extend(top_k_data)

# 创建稀疏化的 A_collab
A_collab = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(n_items, n_items))
print("协同图稀疏化完成。")

# =================================================================================
# ====== 主要修改部分 END =========================================================
# =================================================================================


# --- 4. 加载内容相似性图 A_content (逻辑不变) ---
print(f"正在从 {mm_adj_path} 加载内容相似性图...")
A_content_sparse_gpu = torch.load(mm_adj_path)
A_content_sparse = A_content_sparse_gpu.cpu().coalesce()
A_content_indices = A_content_sparse.indices().numpy()
A_content_values = A_content_sparse.values().numpy()
A_content = sp.coo_matrix((A_content_values, (A_content_indices[0], A_content_indices[1])), shape=(n_items, n_items))
print("内容相似性图 A_content 加载完成。")


# --- 5. 融合图 (逻辑不变) ---
print(f"开始融合图，beta = {beta} ...")
A_fused = beta * A_content + (1 - beta) * A_collab
A_fused = A_fused.tocoo()


# --- 6. 在融合图上执行社区发现 (逻辑不变) ---
print("在融合图上执行Louvain社区发现...")
G = nx.Graph()
G.add_edges_from(zip(A_fused.row, A_fused.col))
partition = community_louvain.best_partition(G)
print("社区发现完成！")


# --- 7. 整理并保存结果 (逻辑不变) ---
item_community_map = torch.zeros(n_items, dtype=torch.long)
for node, community_id in partition.items():
    if node < n_items:
        item_community_map[node] = community_id

torch.save(item_community_map, output_file)

print(f"物品到【融合社区】的映射已保存到: {output_file}")
print("发现的社区总数:", item_community_map.max().item() + 1)
print("张量预览:", item_community_map[:20])