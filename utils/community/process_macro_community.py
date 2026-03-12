import torch
import networkx as nx
from community import community_louvain
import os
import scipy.sparse as sp
import numpy as np
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

# --- 超参数 ---
content_knn_k = 10
collab_knn_k = 10
beta = 0.8

# =================================================================================
# ====== 主要修改部分 START =======================================================
# =================================================================================
# 路径
# 将输出文件名更改，以区分不同的消融实验
output_file = os.path.join(dataset_path, 'item_to_macro_community.pt')
# =================================================================================
# ====== 主要修改部分 END =========================================================
# =================================================================================
mm_adj_path = os.path.join(dataset_path, f'mm_adj_{content_knn_k}.pt')


# =================================================================================
# ====== 阶段一：使用Louvain发现宏观社区 (此部分保持不变) START ================
# =================================================================================
print("--- 阶段一：开始发现宏观社区 ---")
# 1. 加载数据和交互矩阵
dataset = RecDataset(config)
n_users, n_items = dataset.user_num, dataset.item_num
rows = dataset.df[config['USER_ID_FIELD']].values
cols = dataset.df[config['ITEM_ID_FIELD']].values
interaction_matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))

# 2. 计算并稀疏化协同图
print("计算物品间的协同余弦相似度...")
A_collab_dense = cosine_similarity(interaction_matrix.transpose(), dense_output=False).tocsr()
A_collab_dense.setdiag(0)
print("协同相似性图 A_collab 构建完成。")

print(f"正在对协同图进行 Top-K 稀疏化, K={collab_knn_k} ...")
n_items_val, _ = A_collab_dense.shape
new_rows, new_cols, new_data = [], [], []
for i in range(n_items_val):
    start_ptr = A_collab_dense.indptr[i]
    end_ptr = A_collab_dense.indptr[i+1]
    if start_ptr == end_ptr: continue
    row_indices = A_collab_dense.indices[start_ptr:end_ptr]
    row_data = A_collab_dense.data[start_ptr:end_ptr]
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
A_collab = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(n_items, n_items))
print("协同图稀疏化完成。")

# 3. 加载内容图并融合
A_content_sparse = torch.load(mm_adj_path).cpu().coalesce()
A_content_indices = A_content_sparse.indices().numpy()
A_content_values = A_content_sparse.values().numpy()
A_content = sp.coo_matrix((A_content_values, (A_content_indices[0], A_content_indices[1])), shape=(n_items, n_items))
A_fused = beta * A_content + (1 - beta) * A_collab
A_fused = A_fused.tocoo()

# 4. Louvain社区发现
G = nx.Graph()
G.add_edges_from(zip(A_fused.row, A_fused.col))
partition = community_louvain.best_partition(G)
num_macro_communities = len(set(partition.values()))
print(f"阶段一完成！发现 {num_macro_communities} 个宏观社区。")
# =================================================================================
# ====== 阶段一 END ===============================================================
# =================================================================================


# =================================================================================
# ====== 阶段二已被删除 START =======================================================
# =================================================================================
# print("\n--- 阶段二：开始在宏观社区内进行自适应K-Means细分 ---")
# (整个阶段二的逻辑都被删除了)
# =================================================================================
# ====== 阶段二已被删除 END =========================================================
# =================================================================================


# --- 保存最终结果 ---
# 这里的 partition 直接来自于阶段一的输出
item_community_map = torch.zeros(n_items, dtype=torch.long)
for node, community_id in partition.items():
    if node < n_items:
        item_community_map[node] = community_id

torch.save(item_community_map, output_file)

print(f"\n物品到【宏观社区】的映射已保存到: {output_file}")
print("发现的社区总数:", item_community_map.max().item() + 1)
print("张量预览:", item_community_map[:20])