import torch
import os
import numpy as np
from sklearn.cluster import KMeans

# --- 配置 ---
config = {
    'dataset': 'baby',
    'data_path': '/home/ljy/Documents/nwu/science/code/TT2/data/'
}
dataset_name = config['dataset']
dataset_path = os.path.join(config['data_path'], dataset_name)

# --- 新增超参数: K-Means的簇数量 (K值) ---
# 这是一个需要您根据数据集情况调整的关键超参数
# 可以尝试不同的值，例如 100, 200, 500
num_clusters = 400

# =================================================================================
# ====== 主要修改部分 START =======================================================
# =================================================================================

# --- 输入/输出文件路径 ---
# 根据您的反馈，将输入文件名修正为 text_feat.npy
feature_file_path = os.path.join(dataset_path, 'text_feat.npy')
output_file = os.path.join(dataset_path, 'item_to_cluster.pt') # 输出文件名保持不变

# --- 1. 加载用于聚类的特征数据 ---
print(f"正在从 {feature_file_path} 加载物品特征数据...")
if not os.path.exists(feature_file_path):
    print(f"错误: 特征文件未找到 at {feature_file_path}")
    exit()

# 使用 np.load() 来加载 .npy 文件
item_features_np = np.load(feature_file_path)
n_items, feat_dim = item_features_np.shape
print(f"特征数据加载完成: {n_items} 个物品, {feat_dim} 维特征。")

# --- 2. 执行K-Means聚类 ---
print(f"开始对物品特征执行K-Means聚类，K={num_clusters} ...")
# n_init='auto' 是 scikit-learn 的新推荐设置，以避免未来版本中的警告
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto', verbose=1)

# .fit_predict 会直接返回每个物品所属的簇ID
cluster_ids = kmeans.fit_predict(item_features_np)
print("K-Means聚类完成！")

# =================================================================================
# ====== 主要修改部分 END =========================================================
# =================================================================================

# --- 3. 整理并保存结果 (逻辑不变) ---
# 将 NumPy 数组转换为 PyTorch 张量
item_cluster_map = torch.tensor(cluster_ids, dtype=torch.long)

# 保存为 .pt 文件
torch.save(item_cluster_map, output_file)

print(f"物品到【聚类簇】的映射已保存到: {output_file}")
print("聚类簇总数:", num_clusters)
print("张量预览:", item_cluster_map[:20])
print("张量形状:", item_cluster_map.shape)