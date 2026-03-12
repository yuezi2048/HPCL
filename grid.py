import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 准备更新后的 5x5 实验数据（来自您的表格）
lambda_graded_align_values = [0.05, 0.10, 0.20, 0.30, 0.40]
infoNCETemp_values = [0.05, 0.10, 0.20, 0.40, 0.60]

# 表格中的 Recall 数据
recall_data = np.array([
    # lambda_graded_align ->
    # 0.05    0.10    0.20    0.30    0.40
    [0.1040, 0.1028, 0.1037, 0.1040, 0.1035], # infoNCETemp = 0.05
    [0.1042, 0.1036, 0.1038, 0.1034, 0.1040], # infoNCETemp = 0.10
    [0.1041, 0.1038, 0.1042, 0.1028, 0.1020], # infoNCETemp = 0.20
    [0.1032, 0.1031, 0.1032, 0.1026, 0.1019], # infoNCETemp = 0.40
    [0.1026, 0.1032, 0.1038, 0.1040, 0.1017]  # infoNCETemp = 0.60
])

# 为了让Y轴从下往上递增，反转数据和标签的顺序
recall_data_reversed = np.flipud(recall_data)
infoNCETemp_labels_reversed = infoNCETemp_values[::-1]

# 2. 设置图形风格和背景色 (保持不变)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

plt.figure(figsize=(7, 6))

# 3. 创建热力图 (保持风格不变)
sns.heatmap(
    recall_data_reversed,
    annot=True,
    fmt=".4f",
    cmap="Reds", # 保持原有风格
    cbar=False,  # 保持原有风格
    linewidths=1,
    linecolor='white',
    # 自动适应新数据的范围
    vmin=recall_data.min() - 0.0005,
    vmax=recall_data.max() + 0.0005,
    annot_kws={"size": 10, "fontweight": "bold"} # 保持原有风格
)

# 4. 設置軸標籤和標題 (更新为新参数)
plt.xlabel(r'Hyper-parameter $\lambda_{graded\_align}$', fontsize=16, fontweight='bold', labelpad=20)
plt.xticks(np.arange(len(lambda_graded_align_values)) + 0.5, lambda_graded_align_values, fontsize=12)

# 更新 Y 轴标签
plt.ylabel(r'Hyper-parameter $infoNCETemp$', fontsize=16, fontweight='bold', rotation=90, labelpad=20)
plt.yticks(np.arange(len(infoNCETemp_values)) + 0.5, infoNCETemp_labels_reversed, rotation=0, fontsize=12)

# 保持标题，更新副标题
plt.title('Recall@20', fontsize=20, fontweight='bold', pad=20)
plt.text(2, -0.7, r'Baby - ($infoNCETemp, \lambda_{graded\_align}$)', # 更新副标题以匹配新参数
         horizontalalignment='center', verticalalignment='center',
         fontsize=16, fontweight='bold', transform=plt.gca().transData)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图片到文件
output_filename = 'heatmap_baby_infonce_lambda.png' # 更新文件名
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Heatmap has been saved to '{output_filename}'")