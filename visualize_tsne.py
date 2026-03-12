import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import yaml # 导入 PyYAML 库

# --- 假设您的所有模块都位于正确的路径 ---
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer
from models.tt6 import TT6
from common.trainer import Trainer

# ====================================================================
# 1. 从 YAML 文件加载配置
# ====================================================================
print("--- 步骤 1: 从 config.yaml 加载配置 ---")
config_file_path = 'confsconfig.yaml'
try:
    with open(config_file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
except FileNotFoundError:
    print(f"错误: '{config_file_path}' 未找到。请确保配置文件与脚本在同一目录下。")
    exit()

# 使用 Config 类来处理和初始化配置
config = Config(model='TT6', dataset='baby', config_dict=config_dict)
init_seed(config['seed'][0]) # 使用种子列表的第一个值进行初始化
config['device'] = 'cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu'

# 为了快速演示，强制只训练几个epoch
config['epochs'] = 5
print(f"配置加载完毕。将使用设备: {config['device']}")


# ====================================================================
# 2. 加载数据
# ====================================================================
print("\n--- 步骤 2: 加载并分割数据 ---")
dataset = RecDataset(config)
train_dataset, valid_dataset, test_dataset = dataset.split()
train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
print("数据加载完毕。")
n_users, n_items = dataset.user_num, dataset.item_num


# ====================================================================
# 3. 创建并训练 TT6 模型实例
# ====================================================================
print("\n--- 步骤 3: 创建并训练 TT6 模型实例 ---")
model_instance = get_model(config['model'])(config, dataset).to(config['device'])
print("TT6 模型实例已创建。")
print(model_instance)

trainer = get_trainer()(config, model_instance)
print("训练器已创建，开始训练...")

# 开始训练
# 注意：由于配置中包含超参数列表，这里的训练实际上可能只运行一个组合
# 完整的超参数搜索请使用 quick_start.py
best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
    train_data, valid_data=valid_data, test_data=test_data, saved=True
)
print("模型训练完成！")
trained_model = trainer.model


# ====================================================================
# 4. 使用训练好的模型进行 t-SNE 可视化
# ====================================================================
# （此部分与上一版脚本相同，此处省略以保持简洁）
print("\n--- 步骤 4: 开始 t-SNE 可视化 ---")
trained_model.eval()
trained_model.to(torch.device('cpu'))

N_ITEMS_TO_SAMPLE = 500
TSNE_PERPLEXITY = 30
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

with torch.no_grad():
    dummy_interaction = [torch.tensor([0]), torch.tensor([0]), torch.tensor([1])]
    trained_model.forward(dummy_interaction)
    all_item_embeddings = trained_model.result_embed[n_users:].detach().cpu().numpy()
    all_community_labels = trained_model.item_community_map.detach().cpu().numpy()

item_indices = np.arange(n_items)
sampled_indices = np.random.choice(item_indices, N_ITEMS_TO_SAMPLE, replace=False)
sampled_embeddings = all_item_embeddings[sampled_indices]
sampled_communities = all_community_labels[sampled_indices]

tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=1000, random_state=RANDOM_SEED)
embeddings_2d = tsne.fit_transform(sampled_embeddings)

plt.figure(figsize=(14, 12))
sns.set_style("whitegrid")
plot = sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=sampled_communities,
    palette=sns.color_palette("deep", n_colors=len(np.unique(sampled_communities))),
    legend='full', alpha=0.8, s=50
)
if len(np.unique(sampled_communities)) > 20:
    plot.legend_.remove()

plt.title(f't-SNE of {N_ITEMS_TO_SAMPLE} Item Embeddings from Trained TT6 (Colored by Community)', fontsize=18)
plt.xlabel('t-SNE Dimension 1', fontsize=14)
plt.ylabel('t-SNE Dimension 2', fontsize=14)
plt.tight_layout()
output_filename = 'trained_tt6_tsne_visualization.png'
plt.savefig(output_filename, dpi=300)
print(f"\n可视化图像已成功保存为 '{output_filename}'")