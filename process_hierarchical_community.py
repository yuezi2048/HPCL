import torch
import networkx as nx
from community import community_louvain
import os
import scipy.sparse as sp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
# 第一阶段参数
content_knn_k = 10
collab_knn_k = 10
beta = 0.8

# =================================================================================
# ====== 主要修改部分 START =========================================================
# =================================================================================
# 第二阶段参数：不再使用固定的k，而是定义一个范围
min_sub_clusters = 5  # 每个宏观社区最少聚成5个子类
max_sub_clusters = 50  # 每个宏观社区最多聚成50个子类
# =================================================================================
# ====== 主要修改部分 END ===========================================================

# 路径
output_file = os.path.join(dataset_path, 'item_to_hierarchical_community.pt')
mm_adj_path = os.path.join(dataset_path, f'mm_adj_{content_knn_k}.pt')
text_feat_path = os.path.join(dataset_path, 'text_feat.npy')

# =================================================================================
# ====== 阶段一：使用Louvain发现宏观社区 START ===================================
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
    if start_ptr == end_ptr:
        continue
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
partition = community_louvain.best_partition(G, random_state=42) # 42是一个常用的固定整数
num_macro_communities = len(set(partition.values()))
print(f"阶段一完成！发现 {num_macro_communities} 个宏观社区。")
# =================================================================================
# ====== 阶段一 END ===============================================================
# =================================================================================

# =================================================================================
# ====== 阶段二：进行自适应的K-Means细分 START ========================
# =================================================================================
print("\n--- 阶段二：开始在宏观社区内进行自适应K-Means细分 ---")
# 1. 加载用于K-Means的文本特征
text_features = np.load(text_feat_path)

# 2. 将物品按宏观社区分组
communities = {}
for item_id, community_id in partition.items():
    if community_id not in communities:
        communities[community_id] = []
    communities[community_id].append(item_id)

# 3. 准备最终的层级化社区标签
item_hierarchical_community_map = torch.zeros(n_items, dtype=torch.long)
total_clusters = 0

# 4. 遍历每个宏观社区，进行内部聚类
for macro_id, items_in_macro in communities.items():
    num_items_in_macro = len(items_in_macro)

    # --- 核心自适应逻辑 ---
    # 根据平方根法则动态计算k值
    dynamic_k = int(np.sqrt(num_items_in_macro))
    # 将k值限制在预设的最小和最大范围内
    k_for_this_community = max(min_sub_clusters, min(dynamic_k, max_sub_clusters))

    print(f"正在处理宏观社区 {macro_id} (含 {num_items_in_macro} 物品)，动态设定 K = {k_for_this_community}...")

    # 如果社区内物品太少，则不进行细分
    if num_items_in_macro < k_for_this_community:
        for item_id in items_in_macro:
            item_hierarchical_community_map[item_id] = total_clusters
        total_clusters += 1
        continue

    # 提取该社区内所有物品的文本特征
    community_features = text_features[items_in_macro]

    # --- 新增: PCA降维 ---
    # 只有当特征维度较高且物品较多时才执行
    if community_features.shape[1] > 128 and len(items_in_macro) > 128:
        print("    -> 正在执行PCA降维...")
        pca = PCA(n_components=128)  # 降到128维
        community_features = pca.fit_transform(community_features)
    # --- 新增结束 ---

    # 使用动态计算的k值进行K-Means聚类
    kmeans = KMeans(n_clusters=k_for_this_community, random_state=42, n_init='auto')
    sub_labels = kmeans.fit_predict(community_features)

    # 创建全局唯一的层级化标签
    for i, item_id in enumerate(items_in_macro):
        hierarchical_label = total_clusters + sub_labels[i]
        item_hierarchical_community_map[item_id] = hierarchical_label

    # 更新全局总簇数
    total_clusters += k_for_this_community

print("阶段二完成！")
# =================================================================================
# ====== 阶段二 END ===============================================================
# =================================================================================

# --- 保存最终结果 ---
torch.save(item_hierarchical_community_map, output_file)

print(f"\n物品到【层级化社区】的映射已保存到: {output_file}")
print("发现的最终社区总数:", total_clusters)
print("张量预览:", item_hierarchical_community_map[:20])