import torch
import networkx as nx
from community import community_louvain
import os
import scipy.sparse as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 配置 ---
config = {
    'dataset': 'baby',
    'data_path': '/home/ljy/Documents/nwu/science/code/TT2/data/'
}
dataset_name = config['dataset']
dataset_path = os.path.join(config['data_path'], dataset_name)

# --- 超参数 ---
# 1. 融合图像和文本相似度时的权重
content_image_weight = 0.5  # 0.5代表图文同等重要

# 2. 构建k-NN图时的K值
knn_k = 20  # 可以尝试不同的K值, e.g., 10, 20, 50

# --- 输入/输出文件路径 ---
# 根据您的反馈，加载 .npy 格式的原始特征文件
image_feat_path = os.path.join(dataset_path, 'image_feat.npy')
text_feat_path = os.path.join(dataset_path, 'text_feat.npy')
# 输出一个新的社区标签文件，以作区分
output_file = os.path.join(dataset_path, 'item_to_content_community.pt')

# =================================================================================
# ====== 核心逻辑: 基于原始特征构建图并发现社区 START =======================
# =================================================================================

# --- 1. 加载原始特征数据 ---
print(f"正在从 .npy 文件加载原始图像和文本特征...")
if not os.path.exists(image_feat_path) or not os.path.exists(text_feat_path):
    print(f"错误: 特征文件未找到 at {image_feat_path} 或 {text_feat_path}")
    exit()

image_features = np.load(image_feat_path)
text_features = np.load(text_feat_path)
n_items, _ = image_features.shape
print(f"特征加载完成: {n_items} 个物品。")

# --- 2. 计算并融合内容相似度矩阵 ---
print("计算图像特征的余弦相似度...")
sim_image = cosine_similarity(image_features)

print("计算文本特征的余弦相似度...")
sim_text = cosine_similarity(text_features)

print(f"融合图文相似度，图像权重为: {content_image_weight} ...")
# 融合得到一个稠密的相似度矩阵
sim_fused_dense = content_image_weight * sim_image + (1 - content_image_weight) * sim_text
# 确保对角线为0，避免自环
np.fill_diagonal(sim_fused_dense, 0)

# --- 3. 对融合后的内容相似度图进行 Top-K 稀疏化 ---
print(f"正在对纯内容相似度图进行 Top-K 稀疏化, K={knn_k} ...")
new_rows, new_cols, new_data = [], [], []

# 遍历每个物品，找到其Top-K相似的邻居
for i in range(n_items):
    # 获取当前物品与其他所有物品的相似度分数
    row_data = sim_fused_dense[i]

    # 找到Top-K的邻居索引 (使用argpartition高效实现)
    # 如果总物品数小于K，则取所有邻居
    k_neighbors = min(knn_k, n_items - 1)
    top_k_indices = np.argpartition(-row_data, k_neighbors)[:k_neighbors]
    top_k_data = row_data[top_k_indices]

    new_rows.extend([i] * len(top_k_indices))
    new_cols.extend(top_k_indices)
    new_data.extend(top_k_data)

# 创建稀疏化的纯内容图 A_content
A_content_sparse = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(n_items, n_items))
print("纯内容图稀疏化完成。")

# --- 4. 在纯内容图上执行社区发现 ---
print("在纯内容图上执行Louvain社区发现...")
G = nx.Graph()
G.add_edges_from(zip(A_content_sparse.row, A_content_sparse.col))
partition = community_louvain.best_partition(G)
print("社区发现完成！")

# --- 5. 整理并保存结果 ---
item_community_map = torch.zeros(n_items, dtype=torch.long)
for node, community_id in partition.items():
    if node < n_items:
        item_community_map[node] = community_id

torch.save(item_community_map, output_file)

print(f"物品到【纯内容社区】的映射已保存到: {output_file}")
print("发现的社区总数:", item_community_map.max().item() + 1)
print("张量预览:", item_community_map[:20])