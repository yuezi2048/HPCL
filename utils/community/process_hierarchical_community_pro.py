import torch
import igraph as ig                      # <--- 新增
import leidenalg as la                   # <--- 新增
import os
import scipy.sparse as sp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# --- 导入您的 Dataset 类 ---
from utils.dataset import RecDataset

# --- ↓↓↓ 添加下面的代码块 ↓↓↓ ---
import random

# 设置一个全局随机种子以保证所有操作的可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# --- ↑↑↑ 添加结束 ↑↑↑ ---

# --- 配置 ---
config = {
    'dataset': 'baby',
    'data_path': '/root/autodl-tmp/code/TT2/data/',
    # 'data_path': '/home/ljy/Documents/nwu/science/code/TT2/data/',
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

# 3. 加载内容图
A_content_sparse = torch.load(mm_adj_path).cpu().coalesce()
A_content_indices = A_content_sparse.indices().numpy()
A_content_values = A_content_sparse.values().numpy()
A_content = sp.coo_matrix((A_content_values, (A_content_indices[0], A_content_indices[1])), shape=(n_items, n_items))
A_fused = A_content
A_fused = A_fused.tocoo()

# 4. Leiden社区发现 (新版本)
print("在融合图上执行Leiden社区发现...")
# Leiden算法使用 igraph 对象，而不是 networkx
# 从边列表直接创建 igraph 图
edges = list(zip(A_fused.row, A_fused.col))
G_ig = ig.Graph(edges=edges, directed=False)

# 执行Leiden算法
# find_partition 返回一个划分对象
# --- ↓↓↓ 修改这一行 ↓↓↓ ---
# 原代码:
# partition_ig = la.find_partition(G_ig, la.ModularityVertexPartition)
# 修改后:
partition_ig = la.find_partition(G_ig, la.ModularityVertexPartition, seed=SEED)
# --- ↑↑↑ 修改结束 ↑↑↑ ---

# 将结果转换为与Louvain输出兼容的 {节点: 社区ID} 字典格式
partition = {node.index: membership for node, membership in zip(G_ig.vs, partition_ig.membership)}

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
        # --- ↓↓↓ 修改这一行 ↓↓↓ ---
        # 原代码:
        # pca = PCA(n_components=128)
        # 修改后:
        pca = PCA(n_components=128, random_state=SEED)
        # --- ↑↑↑ 修改结束 ↑↑↑ ---
        community_features = pca.fit_transform(community_features)
    # --- 新增结束 ---

    # 使用动态计算的k值进行K-Means聚类
    # --- ↓↓↓ 修改这一行 ↓↓↓ ---
    # 原代码:
    # kmeans = KMeans(n_clusters=k_for_this_community, random_state=42, n_init='auto')
    # 修改后:
    kmeans = KMeans(n_clusters=k_for_this_community, random_state=SEED, n_init='auto')
    # --- ↑↑↑ 修改结束 ↑↑↑ ---
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