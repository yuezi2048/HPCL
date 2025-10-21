import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree


class GCN(nn.Module):
    """一个简化的GCN模型，包含残差连接和噪声注入功能"""

    # Add 'feature_dim' to the constructor
    def __init__(self, num_users, num_items, embedding_dim, aggr_mode, feature_dim, latent_dim=None, device=None):
        super(GCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim or embedding_dim
        self.device = device

        self.user_preferences = nn.Parameter(torch.empty(num_users, self.latent_dim, device=self.device))
        nn.init.xavier_normal_(self.user_preferences.data)

        self.feature_mlp = nn.Sequential(
            # Use 'feature_dim' as the input dimension
            nn.Linear(feature_dim, 4 * self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * self.latent_dim, self.latent_dim)
        )
        self.gcn_layer1 = BaseGCN(self.latent_dim, self.latent_dim, aggr=aggr_mode)

    def forward(self, edge_index, item_features, perturbed=False):
        # 初始物品特征通过MLP进行转换
        transformed_features = self.feature_mlp(item_features)

        # 拼接用户偏好和物品特征作为GCN的输入
        initial_embeddings = torch.cat((self.user_preferences, transformed_features), dim=0)
        initial_embeddings = F.normalize(initial_embeddings, p=2, dim=1)

        # GCN传播
        propagated_embeddings = self.gcn_layer1(initial_embeddings, edge_index)
        if perturbed:
            noise = torch.rand_like(propagated_embeddings, device=self.device)
            propagated_embeddings += torch.sign(propagated_embeddings) * F.normalize(noise, dim=-1) * 0.1

        propagated_embeddings_2 = self.gcn_layer1(propagated_embeddings, edge_index)
        if perturbed:
            noise = torch.rand_like(propagated_embeddings_2, device=self.device)
            propagated_embeddings_2 += torch.sign(propagated_embeddings_2) * F.normalize(noise, dim=-1) * 0.1

        # 应用残差连接
        final_embeddings = initial_embeddings + propagated_embeddings + propagated_embeddings_2
        return final_embeddings, self.user_preferences


class BaseGCN(MessagePassing):
    """基础的GCN层，继承自torch_geometric的MessagePassing"""

    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(BaseGCN, self).__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 处理度为0的节点

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j