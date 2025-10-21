# -*- coding: utf-8 -*-
"""
t-SNE 可视化实验脚本
根据提供的实验方案，通过 t-SNE 可视化来定性分析 TT5 模型解耦模块的有效性。
该脚本会加载一个已训练好的 TT5 模型并生成可视化结果。
"""
import os
import torch
import numpy as np
import yaml  # 导入PyYAML
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')  # 使用 'Agg' 後端，專為在無圖形界面的服務器上保存文件而設計
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# ----------------------------------------------------------------------------------
# 关键：请确保您的 tt5.py 和 utils 文件夹在 Python 路径中
# ----------------------------------------------------------------------------------
from models.tt5 import TT5
# 从您的自定义框架中导入所需模块
from utils.configurator import Config
from utils.dataset import RecDataset

warnings.filterwarnings('ignore', category=UserWarning, message='.*invalid value encountered in T-SNE.*')


# ===============================================================================================
# Part 1 - 模型和数据加载 (Model and Data Loading)
# ===============================================================================================

def load_real_model(model_path, config_file_list, dataset_name):
    """
    加载真实的、已训练好的 TT5 模型检查点 (适配您的自定义框架)。
    """
    print(f"--- 正在加载模型: {model_path} ---")

    # 1. 从 YAML 文件加载配置字典
    config_dict = {}
    if config_file_list:
        for file in config_file_list:
            with open(file, 'r', encoding='utf-8') as f:
                config_dict.update(yaml.safe_load(f))

    # 2. 初始化自定义的 Config 和 Dataset 对象
    config = Config(model='TT5', dataset=dataset_name, config_dict=config_dict)
    dataset = RecDataset(config)

    # 3. 初始化模型
    model = TT5(config, dataset).to(config['device'])

    # 4. 加载模型状态
    checkpoint = torch.load(model_path, map_location=config['device'])
    if 'state_dict' not in checkpoint:
        raise ValueError(f"在模型检查点文件 '{model_path}' 中找不到 'state_dict' 键。请确保您的模型保存了 state_dict。")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("--- 模型加载成功! ---")

    # 5. 计算并缓存所有物品的嵌入向量
    with torch.no_grad():
        print("--- 正在计算所有物品的嵌入向量... ---")
        # 确保您的模型有这个方法来计算所有物品的 G 和 S 特征
        if hasattr(model, 'calculate_embedding'):
            model.calculate_embedding()
        else:
            print(
                "警告: 模型中未找到 'calculate_embedding' 方法。请确保在运行实验前，模型的 item_image_g, item_text_g 等特征已被计算。")

        print("--- 嵌入向量计算完成! ---")

    return model, dataset


# ===============================================================================================
# 实验 3.1: 验证通用特征 (G) 的对齐与聚类效果
# ===============================================================================================
def run_experiment_1(model, dataset):
    print("\n--- 正在运行实验 3.1: 验证通用特征 (G) ---")

    # 1. 选取样本 (Select samples)
    # TODO: 替换为您自己数据集中的物品ID和类别
    # 您需要知道哪些 item_id 属于哪个类别。ID通常从1开始。
    n_samples_per_category = 30
    categories = {
        'Shoes': list(range(1, 1 + n_samples_per_category)),  # 示例: 物品ID 1 到 30 是鞋子
        'Tops': list(range(101, 101 + n_samples_per_category)),  # 示例: 物品ID 101 到 130 是上衣
        'Pants': list(range(201, 201 + n_samples_per_category))  # 示例: 物品ID 201 到 230 是裤子
    }

    all_item_ids = [item_id for sublist in categories.values() for item_id in sublist]
    item_labels = [label for label, ids in categories.items() for _ in ids]

    # 2. 提取特征 (Extract features)
    with torch.no_grad():
        item_image_g = model.item_image_g
        item_text_g = model.item_text_g

    selected_image_g = item_image_g[all_item_ids].cpu().numpy()
    selected_text_g = item_text_g[all_item_ids].cpu().numpy()

    # 3. 降维与可视化 (Dimensionality reduction and visualization)
    features_to_tsne = np.vstack([selected_image_g, selected_text_g])

    print("正在进行 t-SNE 降维 (G)...")
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, n_iter=1000, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_to_tsne)

    tsne_image = tsne_results[:len(all_item_ids)]
    tsne_text = tsne_results[len(all_item_ids):]

    # 4. 绘图 (Plotting)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = {'Shoes': '#1f77b4', 'Tops': '#ff7f0e', 'Pants': '#2ca02c'}

    for i, label in enumerate(item_labels):
        ax.scatter(tsne_image[i, 0], tsne_image[i, 1], color=colors[label], marker='o', s=80, alpha=0.7)
        ax.scatter(tsne_text[i, 0], tsne_text[i, 1], color=colors[label], marker='x', s=80, alpha=0.7)
        ax.plot([tsne_image[i, 0], tsne_text[i, 0]], [tsne_image[i, 1], tsne_text[i, 1]],
                color=colors[label], linestyle='-', linewidth=0.5, alpha=0.6)

    legend_elements = [
                          Line2D([0], [0], marker='o', color='w', label='Image Feature (G)', markerfacecolor='gray',
                                 markersize=10),
                          Line2D([0], [0], marker='x', color='w', label='Text Feature (G)', markeredgecolor='gray',
                                 markersize=10),
                      ] + [Line2D([0], [0], color=color, lw=4, label=label) for label, color in colors.items()]

    ax.legend(handles=legend_elements, title="Legend", fontsize=12)
    ax.set_title("t-SNE of General Features (G) by Category", fontsize=16, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    save_path = "experiment_1_general_features.png"
    plt.savefig(save_path, dpi=300)
    print(f"实验 3.1 完成，结果已保存至 '{save_path}'")
    plt.close(fig)  # 关闭图形，释放内存


# ===============================================================================================
# 实验 3.2: 验证特定特征 (S) 的分离效果
# ===============================================================================================
def run_experiment_2(model, dataset):
    print("\n--- 正在运行实验 3.2: 验证特定特征 (S) ---")

    # 1. 选取样本 (Select samples)
    # TODO: 替换为您自己数据集中的物品ID和细粒度属性
    n_samples_per_sub_category = 15
    sub_categories = {
        'Plaid Shirt': list(range(101, 101 + n_samples_per_sub_category)),
        'Striped T-shirt': list(range(131, 131 + n_samples_per_sub_category)),
        'White Shirt': list(range(161, 161 + n_samples_per_sub_category))
    }

    all_item_ids = [item_id for sublist in sub_categories.values() for item_id in sublist]
    item_labels = [label for label, ids in sub_categories.items() for _ in ids]

    # 2. 提取特征 (Extract features)
    with torch.no_grad():
        item_image_g = model.item_image_g
        item_image_s = model.item_image_s

    selected_g_features = item_image_g[all_item_ids].cpu().numpy()
    selected_s_features = item_image_s[all_item_ids].cpu().numpy()

    # 3. 降维 (Dimensionality reduction)
    print("正在进行 t-SNE 降维 (G & S)...")
    tsne_g = TSNE(n_components=2, perplexity=10, random_state=42, n_iter=1000, init='pca',
                  learning_rate='auto').fit_transform(selected_g_features)
    tsne_s = TSNE(n_components=2, perplexity=10, random_state=42, n_iter=1000, init='pca',
                  learning_rate='auto').fit_transform(selected_s_features)

    # 4. 绘图 (Plotting)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    colors = {'Plaid Shirt': '#1f77b4', 'Striped T-shirt': '#ff7f0e', 'White Shirt': '#2ca02c'}

    # 子图 1: G 特征的可视化
    for label, color in colors.items():
        indices = [i for i, l in enumerate(item_labels) if l == label]
        ax1.scatter(tsne_g[indices, 0], tsne_g[indices, 1], c=color, label=label, s=100, alpha=0.8)
    ax1.set_title("t-SNE of General Features (G) of 'Tops'", fontsize=16, fontweight='bold')
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax1.legend(title="Sub-categories")
    ax1.grid(True)

    # 子图 2: S 特征的可视化
    for label, color in colors.items():
        indices = [i for i, l in enumerate(item_labels) if l == label]
        ax2.scatter(tsne_s[indices, 0], tsne_s[indices, 1], c=color, label=label, s=100, alpha=0.8)
    ax2.set_title("t-SNE of Specific Features (S) of 'Tops'", fontsize=16, fontweight='bold')
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax2.legend(title="Sub-categories")
    ax2.grid(True)

    plt.tight_layout()
    save_path = "experiment_2_specific_features.png"
    plt.savefig(save_path, dpi=300)
    print(f"实验 3.2 完成，结果已保存至 '{save_path}'")
    plt.close(fig)  # 关闭图形，释放内存


if __name__ == '__main__':
    # ===========================================================================================
    # 主程序入口 (Main execution block)
    # ===========================================================================================

    # TODO: 1. 设置您的模型和配置文件路径
    MODEL_CHECKPOINT_PATH = 'saved/TT5-Oct-11-2025_00-00-00.pth'  # 示例: 'saved/TT5-Oct-10-2025_11-12-00.pth'
    CONFIG_FILES = ['TT5.yaml']  # 您的模型配置文件
    DATASET_NAME = 'baby'  # 您使用的数据集名称，例如 'baby'

    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"错误: 模型文件未找到 '{MODEL_CHECKPOINT_PATH}'")
        print("请在 'main' 代码块中设置正确的模型检查点路径和数据集名称。")
    else:
        # 加载真实模型和数据
        model, dataset = load_real_model(MODEL_CHECKPOINT_PATH, CONFIG_FILES, DATASET_NAME)

        # 运行实验一
        run_experiment_1(model, dataset)

        # 运行实验二
        run_experiment_2(model, dataset)

