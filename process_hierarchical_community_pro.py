import argparse

import torch
import igraph as ig
import leidenalg as la
import os
import scipy.sparse as sp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler  # <-- 优化1: 改成更鲁棒的 scaler
from sklearn.metrics import silhouette_score   # <-- 优化4: 新增量化评估
from collections import Counter
import random

from utils.dataset import RecDataset

# --- 全局可复现种子 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description='层级社区发现脚本')
parser.add_argument('--dataset', '-d', type=str, default='baby', help='数据集名称 (baby/clothing/sports)')
parser.add_argument('--datapath', '-p', type=str, default='./data/', help='数据路径')
args = parser.parse_args()

# --- 配置 ---
config = {
    'dataset': args.dataset,
    'data_path': args.datapath,
    'USER_ID_FIELD': 'userID',
    'ITEM_ID_FIELD': 'itemID',
    'inter_splitting_label': 'x_label',
    'field_separator': '\t',
    'filter_out_cod_start_users': False
}
dataset_name = config['dataset']
dataset_path = os.path.join(config['data_path'], dataset_name)
config['inter_file_name'] = f'{dataset_name}.inter'

# --- 超参数（已优化）---
content_knn_k = 10
collab_knn_k = 10
alpha = 0.4                    # <-- 优化2: collab 权重提升到 0.4
resolution_parameter = 0.5     # <-- 优化2: 减少碎社区
min_sub_clusters = 5
max_sub_clusters = 20

# --- 路径 ---
output_file = os.path.join(dataset_path, 'item_to_hierarchical_community.pt')
macro_file = os.path.join(dataset_path, 'macro_community.pt')  # <-- 新增中间结果
mm_adj_path = os.path.join(dataset_path, f'mm_adj_{content_knn_k}.pt')
text_feat_path = os.path.join(dataset_path, 'text_feat.npy')
image_feat_path = os.path.join(dataset_path, 'image_feat.npy')

print("=== 优化版层级社区发现开始 ===")

# =================================================================================
# ====== 阶段一：Leiden 宏观社区（融合协同+内容 + 量化评估） ========================
# =================================================================================
print("阶段一：构建融合图 & Leiden 宏观社区...")
dataset = RecDataset(config)
n_users, n_items = dataset.user_num, dataset.item_num
interaction_matrix = sp.csr_matrix(
    (np.ones(len(dataset.df)),
     (dataset.df[config['USER_ID_FIELD']].values,
      dataset.df[config['ITEM_ID_FIELD']].values)),
    shape=(n_users, n_items))

# 协同图（导师建议5）
A_collab = interaction_matrix.T @ interaction_matrix
A_collab.setdiag(0)
A_collab.eliminate_zeros()
collab_row_sum = np.array(A_collab.sum(axis=1)).flatten()
collab_row_sum[collab_row_sum == 0] = 1.0
A_collab = A_collab.multiply(1.0 / collab_row_sum[:, np.newaxis])

# 内容图 + 行归一化（优化1）
A_content_sparse = torch.load(mm_adj_path).cpu().coalesce()
A_content = sp.coo_matrix(
    (A_content_sparse.values().numpy(),
     (A_content_sparse.indices()[0].numpy(), A_content_sparse.indices()[1].numpy())),
    shape=(n_items, n_items))
content_row_sum = np.array(A_content.sum(axis=1)).flatten()
content_row_sum[content_row_sum == 0] = 1.0
A_content = A_content.multiply(1.0 / content_row_sum[:, np.newaxis])

# 融合（优化2）
A_fused = (1 - alpha) * A_content + alpha * A_collab
A_fused = A_fused.tocoo()

# Leiden
G_ig = ig.Graph(n=n_items, edges=list(zip(A_fused.row, A_fused.col)), directed=False)
partition_ig = la.find_partition(
    G_ig, la.RBConfigurationVertexPartition,
    weights=A_fused.data.tolist(),
    resolution_parameter=resolution_parameter,
    seed=SEED
)
partition = {v.index: m for v, m in zip(G_ig.vs, partition_ig.membership)}
num_macro = len(set(partition.values()))

# === 新增量化评估（优化4）===
modularity = partition_ig.modularity
macro_sizes = sorted(Counter(partition.values()).values())
print(f"阶段一完成！发现 {num_macro} 个宏观社区")
print(f"Modularity（越高越好）: {modularity:.4f}")
print(f"宏社区大小统计: min={min(macro_sizes)}, max={max(macro_sizes)}, avg={np.mean(macro_sizes):.1f}")

# =================================================================================
# ====== 阶段二：自适应 K-Means 子社区（RobustScaler + elbow+silhouette） ============
# =================================================================================
print("\n阶段二：宏观社区内自适应细分...")
text_features = np.load(text_feat_path)
if os.path.exists(image_feat_path):
    image_features = np.load(image_feat_path)
    text_features = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
    image_features = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)
    fused_features = np.concatenate([text_features, image_features], axis=1)
else:
    fused_features = text_features

communities = {}
for item_id, cid in partition.items():
    communities.setdefault(cid, []).append(item_id)

item_hierarchical_community_map = torch.zeros(n_items, dtype=torch.long)
total_clusters = 0
sil_scores = []  # <-- 新增量化评估

for macro_id, items_in_macro in communities.items():
    n = len(items_in_macro)
    if n < min_sub_clusters:
        for item_id in items_in_macro:
            item_hierarchical_community_map[item_id] = total_clusters
        total_clusters += 1
        continue

    # 特征预处理（优化1）
    feats = fused_features[items_in_macro]
    feats = RobustScaler().fit_transform(feats)   # <-- 改成 RobustScaler

    # PCA（保留 85% 方差）
    if feats.shape[1] > 32 and n > 50:
        pca = PCA(n_components=0.85, random_state=SEED)
        feats = pca.fit_transform(feats)

    # 自适应 K（优化3）
    k_cand = list(range(max(2, min_sub_clusters), min(max_sub_clusters + 1, n)))
    if len(k_cand) <= 2:
        best_k = k_cand[0]
    else:
        # 肘部法
        inertias = [KMeans(n_clusters=k, random_state=SEED, n_init=20, init='k-means++').fit(feats).inertia_
                    for k in k_cand]
        p1, p2 = np.array([k_cand[0], inertias[0]]), np.array([k_cand[-1], inertias[-1]])
        dists = [np.abs(np.cross(p2 - p1, np.array([k_cand[i], inertias[i]]) - p1)) / np.linalg.norm(p2 - p1)
                 for i in range(len(inertias))]
        elbow_k = k_cand[np.argmax(dists)]

        # silhouette 精炼（优化3：鲁棒性提升）
        best_k, best_sil = elbow_k, -1
        for dk in [-1, 0, 1]:
            kk = elbow_k + dk
            if kk < 2 or kk > n: continue
            km = KMeans(n_clusters=kk, random_state=SEED, n_init=20, init='k-means++').fit(feats)
            labels = km.labels_
            if len(set(labels)) > 1:
                sil = silhouette_score(feats, labels)
                if sil > best_sil:
                    best_sil, best_k = sil, kk
        if best_sil > 0:
            sil_scores.append(best_sil)

    # 最终聚类
    kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=20, init='k-means++')
    sub_labels = kmeans.fit_predict(feats)
    for i, item_id in enumerate(items_in_macro):
        item_hierarchical_community_map[item_id] = total_clusters + sub_labels[i]
    total_clusters += best_k

# =================================================================================
# ====== 保存 & 最终评估（优化4） =================================================
# =================================================================================
torch.save(item_hierarchical_community_map, output_file)
torch.save(torch.tensor([partition.get(i, 0) for i in range(n_items)]), macro_file)

print(f"\n=== 优化完成！===")
print(f"最终子社区总数: {total_clusters}")
print(f"平均 silhouette 分数: {np.mean(sil_scores):.4f}（越高越好）")
print(f"层级映射已保存: {output_file}")
print(f"宏观分区已保存: {macro_file}")
print(f"张量预览: {item_hierarchical_community_map[:20]}")

# 子社区大小统计（方便继续调参）
sub_sizes = np.bincount(item_hierarchical_community_map.numpy())
print(f"子社区大小统计: min={sub_sizes.min()}, max={sub_sizes.max()}, avg={sub_sizes.mean():.1f}")

# =================================================================================
# ====== 修复版后处理：同宏观内合并 + 全局ID重映射（解决两大隐患） ================
# =================================================================================
print("\n后处理：【同宏观内】合并 size <= 2 的孤立子社区（保留层级边界）...")

from collections import defaultdict

MIN_CLUSTER_SIZE = 3          # ← 可调：想更严格就改成 4
small_threshold = MIN_CLUSTER_SIZE - 1

# 1. 预计算所有子社区中心（全局一次）
cluster_centers = {}
for cid in range(total_clusters):
    mask = (item_hierarchical_community_map == cid)
    if mask.sum() > 0:
        cluster_centers[cid] = fused_features[mask.numpy()].mean(axis=0)

# 2. 按宏观社区分组当前子社区（关键！限制合并范围）
macro_to_subs = defaultdict(set)
for item_id in range(n_items):
    macro_id = partition.get(item_id, 0)          # 使用阶段一的 Leiden 宏观ID
    sub_id = item_hierarchical_community_map[item_id].item()
    macro_to_subs[macro_id].add(sub_id)

reassigned = 0
merged_log = []

for macro_id, sub_ids_in_macro in macro_to_subs.items():
    small_subs = [s for s in sub_ids_in_macro if sub_sizes[s] <= small_threshold]
    if not small_subs:
        continue

    healthy_subs = [s for s in sub_ids_in_macro if sub_sizes[s] >= MIN_CLUSTER_SIZE]

    if healthy_subs:
        # 正常情况：有健康簇 → 把小簇合并到最近的健康簇
        for small_cid in small_subs:
            items_in_small = torch.where(item_hierarchical_community_map == small_cid)[0].numpy()
            if len(items_in_small) == 0:
                continue

            small_center = fused_features[items_in_small].mean(axis=0) if len(items_in_small) > 1 \
                           else fused_features[items_in_small[0]]

            distances = {}
            for h_cid in healthy_subs:
                if h_cid == small_cid:
                    continue
                dist = np.linalg.norm(small_center - cluster_centers[h_cid])
                distances[h_cid] = dist

            if not distances:
                continue

            nearest_cid = min(distances, key=distances.get)

            for item_idx in items_in_small:
                item_hierarchical_community_map[item_idx] = nearest_cid
            reassigned += len(items_in_small)

            sub_sizes[small_cid] = 0
            sub_sizes[nearest_cid] += len(items_in_small)

            merged_log.append(f"宏观 {macro_id}：子簇 {small_cid}(size={len(items_in_small)}) → {nearest_cid}")

    else:
        # 极端情况：没有健康簇，全是小簇
        if len(small_subs) == 1:
            merged_log.append(f"宏观 {macro_id} 只有一个子簇（size={sub_sizes[small_subs[0]]}），保留")
        else:
            target_cid = small_subs[0]
            total_items_merged = 0

            for small_cid in small_subs[1:]:
                items_in_small = torch.where(item_hierarchical_community_map == small_cid)[0].numpy()
                if len(items_in_small) == 0:
                    continue
                for item_idx in items_in_small:
                    item_hierarchical_community_map[item_idx] = target_cid
                total_items_merged += len(items_in_small)
                sub_sizes[small_cid] = 0
                sub_sizes[target_cid] += len(items_in_small)

            merged_log.append(
                f"宏观 {macro_id} 全是小簇（共 {len(small_subs)} 个），已强制合并到子簇 {target_cid}，"
                f"新增 {total_items_merged} 个物品"
            )

print(f"合并完成！共重新分配 {reassigned} 个物品")
if merged_log:
    print("合并详情（前5条）：")
    for line in merged_log[:5]:
        print("  ", line)
    if len(merged_log) > 5:
        print(f"  ... 共 {len(merged_log)} 个合并操作")

# 3. 全局重映射：消除 ID 断层（工程安全）
final_labels_np = item_hierarchical_community_map.numpy()
unique_coms = np.sort(np.unique(final_labels_np))
remap_dict = {old: new for new, old in enumerate(unique_coms)}

item_hierarchical_community_map = torch.tensor([remap_dict[l] for l in final_labels_np], dtype=torch.long)

new_total_clusters = len(unique_coms)
print(f"后处理后有效子社区数量: {new_total_clusters}（ID 已连续 0~{new_total_clusters-1}）")

# 更新大小统计
sub_sizes = np.bincount(item_hierarchical_community_map.numpy())
print(f"最终子社区大小统计: min={sub_sizes.min()}（已无 size=1/2）, max={sub_sizes.max()}, avg={sub_sizes.mean():.1f}")