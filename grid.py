import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 准备更新后的 5x5 精细搜索数据
lambda_graded_align_values = [0.1, 0.2, 0.3, 0.4, 0.5]
alpha_g_s_values = [0.6, 0.7, 0.8, 0.9, 1.0]

recall_20_data = np.array([
    # lambda_graded_align ->
    # 0.1     0.2     0.3     0.4     0.5
    [0.1029, 0.1029, 0.1023, 0.1026, 0.1018], # alpha_g_s = 0.6
    [0.1026, 0.1029, 0.1027, 0.1027, 0.1019], # alpha_g_s = 0.7
    [0.1016, 0.1029, 0.1028, 0.1012, 0.1021], # alpha_g_s = 0.8
    [0.1026, 0.1035, 0.1018, 0.1031, 0.1019], # alpha_g_s = 0.9
    [0.1014, 0.1031, 0.1026, 0.1026, 0.1030]  # alpha_g_s = 1.0
])

# 为了让Y轴从下往上递增，反转数据和标签的顺序
recall_20_data_reversed = np.flipud(recall_20_data)
alpha_g_s_labels_reversed = alpha_g_s_values[::-1]

# 2. 设置图形风格和背景色
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

plt.figure(figsize=(7, 6))

# 3. 创建热力图
sns.heatmap(
    recall_20_data_reversed,
    annot=True,
    fmt=".4f",
    cmap="Reds",
    cbar=False,
    linewidths=1,
    linecolor='white',
    vmin=recall_20_data.min() - 0.0005,
    vmax=recall_20_data.max() + 0.0005,
    annot_kws={"size": 10, "fontweight": "bold"}
)

# 4. 設置軸標籤和標題
plt.xlabel(r'Hyper-parameter $\lambda_{graded\_align}$', fontsize=16, fontweight='bold', labelpad=20)
plt.xticks(np.arange(len(lambda_graded_align_values)) + 0.5, lambda_graded_align_values, fontsize=12)

plt.ylabel(r'Hyper-parameter $\alpha_{g\_s}$', fontsize=16, fontweight='bold', rotation=90, labelpad=20)
plt.yticks(np.arange(len(alpha_g_s_values)) + 0.5, alpha_g_s_labels_reversed, rotation=0, fontsize=12)

plt.title('Recall@20', fontsize=20, fontweight='bold', pad=20)
plt.text(2, -0.7, r'(a) Baby - ($\alpha_{g_s}, \lambda_{graded_align}$)',
         horizontalalignment='center', verticalalignment='center',
         fontsize=16, fontweight='bold', transform=plt.gca().transData)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图片到文件
output_filename = 'heatmap_baby_updated_5x5.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Heatmap has been saved to '{output_filename}'")