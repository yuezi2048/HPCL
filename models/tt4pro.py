# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GCN import GCN
from common.abstract_recommender import GeneralRecommender
from utils.mi_estimator import CLUBSample

class TT4PRO(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TT4PRO, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        # self.construction = 'weighted_max'
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.mask_weight_g = config['mask_weight_g']
        self.mask_weight_f = config['mask_weight_f']
        self.infoNCETemp = config['infoNCETemp']
        self.drop_rate = 0.1
        self.lambda_graded_align = config['lambda_graded_align'] # 兼容旧配置
        self.alpha_g_s = config['alpha_g_s']

        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.mm_adj = None

        self.mlp = nn.Linear(2*dim_x, 2*dim_x)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # --- 核心改动: 加载图社区标签 ---
        # 提示: 'item_to_community.pt' 是您通过图社区发现脚本生成的
        # K-Means
        # community_map_path = os.path.join(dataset_path, 'item_to_cluster.pt ')
        community_map_path = os.path.join(dataset_path, 'item_to_hierarchical_community.pt')
        if os.path.exists(community_map_path):
            item_community_map = torch.load(community_map_path).long()
            self.register_buffer('item_community_map', item_community_map)
        else:
            # 如果文件不存在，创建一个随机的作为占位符，并打印警告
            print("警告: 'item_to_community.pt' 文件未找到。将使用随机社区标签。请运行预处理脚本生成该文件。")
            item_community_map = torch.randint(0, 100, (self.n_items,)).long()
            self.register_buffer('item_community_map', item_community_map)
        # --- 改动结束 ---

        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)

        # self.id_feat = nn.Parameter(
        #     nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
        #                                         requires_grad=True), gain=1).to(self.device))
        # self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
        #                   dim_latent=64, device=self.device, features=self.id_feat)
        #
        # self.result_embed_n1 = nn.Parameter(
        #     nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        # self.result_embed_n2 = nn.Parameter(
        #     nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        # =================================================================
        # DGMRec 解耦模块 (简化版) START
        # =================================================================

        # DGMRec 使用其 'embedding_size' 作为核心维度，我们将其映射到 MENTOR 的 dim_x
        self.embedding_dim = dim_x

        # 1. DGMRec 编码器 (用于分解 G 和 S)
        # (使用 MENTOR 的 self.v_feat 和 self.t_feat 作为输入维度)
        self.image_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # 特定(Specific)特征编码器
        self.image_encoder_s = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder_s = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder_s.weight)
        nn.init.xavier_uniform_(self.text_encoder_s.weight)

        # 2. DGMRec 超参数
        self.lambda_1 = config['lambda_1']

        self.act_g = nn.Tanh()

        # 3. DGMRec MI 估计器 (CLUB)
        self.init_mi_estimator()  # 调用辅助函数

        # =================================================================
        # DGMRec 解耦模块 (简化版) END
        # =================================================================

        # 在 __init__ 尾部添加
        num_communities = torch.max(self.item_community_map).item() + 1
        self.register_buffer('global_prototypes_v', torch.zeros(num_communities, self.embedding_dim))
        self.register_buffer('global_prototypes_t', torch.zeros(num_communities, self.embedding_dim))


    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

        # 【新增】1. 预计算全局原型 (无梯度)
        with torch.no_grad():
            _, _, item_image_s, item_text_s = self.mge()
            num_communities = self.global_prototypes_v.shape[0]
            for m_feat, proto_buffer in zip([item_image_s, item_text_s],
                                            [self.global_prototypes_v, self.global_prototypes_t]):
                sum_feats = torch.zeros(num_communities, self.embedding_dim, device=self.device)
                sum_feats.index_add_(0, self.item_community_map, m_feat)
                counts = torch.bincount(self.item_community_map, minlength=num_communities).unsqueeze(1).clamp(min=1)
                proto_buffer.copy_(F.normalize(sum_feats / counts, dim=1))

        # =================================================================
        # DGMRec MI 估计器训练（保持 3 次更高效）
        # =================================================================
        with torch.no_grad():
            item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        for _ in range(3):  # ← 从5改成3，节省时间
            self.item_image_estimator.train()
            self.item_text_estimator.train()

            item_rand_idx = torch.randperm(self.n_items)[:2048].to(self.device)
            loss_mi = (self.item_image_estimator.learning_loss(item_image_s[item_rand_idx],
                                                               item_image_g[item_rand_idx]) +
                       self.item_text_estimator.learning_loss(item_text_s[item_rand_idx], item_text_g[item_rand_idx]))

            self.optimizer_club.zero_grad()
            loss_mi.backward(retain_graph=True)
            self.optimizer_club.step()

        self.item_image_estimator.eval()
        self.item_text_estimator.eval()

        # =================================================================
        # 【修复】每次 epoch 重新随机 drop 节点（视图增强）
        # =================================================================
        drop_item = torch.randperm(self.n_items, device=self.device)[:int(self.n_items * self.drop_rate)]
        single_cnt = int(len(drop_item) * self.single_percent)
        drop_item_single = drop_item[:single_cnt]

        self.dropv_node_idx_single = drop_item_single[:len(drop_item_single) // 3]
        self.dropt_node_idx_single = drop_item_single[len(drop_item_single) // 3: 2 * len(drop_item_single) // 3]

        # 重新构建 mask（复用 __init__ 中的逻辑，向量化加速）
        mask_cnt = torch.zeros(self.num_item, dtype=torch.long, device=self.device)

        # 【核心修复】：提取出起点为 Item 的所有 ID，并映射回 0~num_item-1 进行度数累加
        items_in_edges = self.edge_index[0][self.edge_index[0] >= self.num_user]
        mask_cnt.index_add_(0, items_in_edges - self.num_user, torch.ones_like(items_in_edges))

        mask_dropv = []
        mask_dropt = []
        dropv_set = set(self.dropv_node_idx_single.cpu().tolist())
        dropt_set = set(self.dropt_node_idx_single.cpu().tolist())

        for idx, num in enumerate(mask_cnt.cpu().tolist()):
            if idx in dropv_set:
                mask_dropv.extend([False] * num)
            else:
                mask_dropv.extend([True] * num)
            if idx in dropt_set:
                mask_dropt.extend([False] * num)
            else:
                mask_dropt.extend([True] * num)

        # =================================================================
        # 【修复】：必须只截取前一半的“单向原始边”进行掩码过滤
        # =================================================================
        half_size = self.edge_index.shape[1] // 2
        edge_index_single = self.edge_index[:, :half_size].cpu().numpy().T  # 形状恢复为 (118551, 2)

        # 按 item 排序重建 edge_index_drop*
        edge_index_single = edge_index_single[np.lexsort(edge_index_single.T[1, None])]
        edge_index_dropv = edge_index_single[mask_dropv]
        edge_index_dropt = edge_index_single[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        # 过滤完成后，再次拼接反向边，恢复模型的双向图要求
        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

    # =================================================================
    # DGMRec MI 估计器训练 END
    # =================================================================

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def InfoNCE(self, view1, view2, temp):
        # 【修复】使用 PyTorch 底层优化的 cross_entropy 实现 InfoNCE
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        sim_matrix = torch.matmul(view1, view2.transpose(0, 1)) / temp
        labels = torch.arange(view1.shape[0], device=self.device)
        return F.cross_entropy(sim_matrix, labels)

    # =================================================================
    # DGMRec 辅助函数 (简化版) START
    # =================================================================

    def init_weight(self, layer):
        # (来自 dgmrec.py)
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def init_mi_estimator(self):
        # (来自 dgmrec.py, 仅初始化物品端的估计器)
        self.item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)
        self.item_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)

        # 将 CLUB 模块的参数收集到一个列表中，以便创建专用的优化器
        params = list(self.item_image_estimator.parameters()) + \
                 list(self.item_text_estimator.parameters())

        # 为 MI 估计器 (CLUB) 创建单独的优化器
        self.optimizer_club = torch.optim.Adam(params, lr=1e-4)

    def mge(self):
        # 模态嵌入 (G 和 S) (来自 dgmrec.py)
        # (使用 MENTOR 继承的 self.image_embedding 和 self.text_embedding)

        # 通用(General)特征 G
        item_image_g = torch.sigmoid(self.shared_encoder(self.act_g(self.image_encoder(self.image_embedding.weight))))
        item_text_g = torch.sigmoid(self.shared_encoder(self.act_g(self.text_encoder(self.text_embedding.weight))))

        # 特定(Specific)特征 S
        item_image_s = torch.sigmoid(self.image_encoder_s(self.image_embedding.weight))
        item_text_s = torch.sigmoid(self.text_encoder_s(self.text_embedding.weight))
        return item_image_g, item_text_g, item_image_s, item_text_s

        # =================================================================
        # DGMRec 辅助函数 (简化版) END
        # =================================================================

    def _generate_representations(self, v_rep, t_rep):
        # 封装的辅助方法，用于从 GCN 输出生成最终嵌入

        # 1. 拼接模态
        representation = torch.cat((v_rep, t_rep), dim=1)

        # 2. 生成用户表征
        # (注意：这里需要对传入的 v_rep, t_rep 进行 unsqueeze)
        v_rep_u = torch.unsqueeze(v_rep[:self.num_user], 2)
        t_rep_u = torch.unsqueeze(t_rep[:self.num_user], 2)
        user_rep = torch.cat((v_rep_u, t_rep_u), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        # 3. 生成并细化物品表征
        item_rep = representation[self.num_user:]
        h = self.buildItemGraph(item_rep)
        item_rep = item_rep + h

        # 4. 组合成最终嵌入
        result_embed = torch.cat((user_rep, item_rep), dim=0)

        # 5. 返回所有需要的结果 (result_embed 用于 BPR, user/item_rep 用于 mask_f_loss)
        return result_embed, user_rep, item_rep

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]

        # 1. 生成所有 GCN 输出
        # 主路径
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)

        # 噪声视图 1
        v_rep_n1, _ = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        t_rep_n1, _ = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)

        # 噪声视图 2
        v_rep_n2, _ = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        t_rep_n2, _ = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)

        # 2. 调用辅助方法来生成最终嵌入
        # 主嵌入 (需要保存 user_rep 和 item_rep 用于 mask_f_loss)
        self.result_embed, self.user_rep, self.item_rep = self._generate_representations(self.v_rep, self.t_rep)

        # 噪声嵌入
        self.result_embed_n1, _, _ = self._generate_representations(v_rep_n1, t_rep_n1)
        self.result_embed_n2, _, _ = self._generate_representations(v_rep_n2, t_rep_n2)

        # 3. 计算分数用于 BPR 损失
        # 【修复】使用局部变量计算偏移，绝对不要修改传入的 pos_item_nodes
        pos_item_nodes_shifted = pos_item_nodes + self.n_users
        neg_item_nodes_shifted = neg_item_nodes + self.n_users
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes_shifted]
        neg_item_tensor = self.result_embed[neg_item_nodes_shifted]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    # def _find_graded_samples(self, batch_items, item_image_g, item_text_g, item_image_s, item_text_s, k=10):
    #     """
    #     为批次中的每个物品动态识别分级样本。
    #     - 强正样本 (R): 通用特征在两个模态上都相似。
    #     - 弱正样本 (T_v, T_t): 特定特征在一个模态上相似，在另一个模态上不相似。
    #     """
    #     # 1. 提取当前批次的物品特征并进行归一化
    #     batch_size = len(batch_items)
    #     device = batch_items.device
    #
    #     g_v = F.normalize(item_image_g[batch_items], dim=1)
    #     g_t = F.normalize(item_text_g[batch_items], dim=1)
    #     s_v = F.normalize(item_image_s[batch_items], dim=1)
    #     s_t = F.normalize(item_text_s[batch_items], dim=1)
    #
    #     # 2. 计算所有相似度矩阵
    #     sim_g_v = g_v @ g_v.T
    #     sim_g_t = g_t @ g_t.T
    #     sim_s_v = s_v @ s_v.T
    #     sim_s_t = s_t @ s_t.T
    #
    #     # 创建一个对角线为-inf的掩码，防止物品将自己选为邻居
    #     self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    #
    #     # 3. 识别强正样本 (Multi-modal Similar, R)
    #     score_multi = sim_g_v * sim_g_t
    #     score_multi.masked_fill_(self_mask, -torch.inf)
    #     _, R_indices = torch.topk(score_multi, k=min(k, batch_size - 1), dim=1)
    #
    #     # 4. 识别弱正样本 (Single-modal Similar, T_v, T_t)
    #     # 创建一个掩码，排除已经被选为强正样本的物品
    #     R_mask = torch.zeros_like(score_multi, dtype=torch.bool).scatter_(1, R_indices, True)
    #     R_mask.logical_or_(self_mask)  # 同时也排除自己
    #
    #     # 视觉弱正样本 T_v
    #     score_single_v = sim_s_v * (1 - sim_s_t)
    #     score_single_v.masked_fill_(R_mask, -torch.inf)
    #     _, T_v_indices = torch.topk(score_single_v, k=min(k, batch_size - 1 - k), dim=1)
    #
    #     # 文本弱正样本 T_t
    #     score_single_t = sim_s_t * (1 - sim_s_v)
    #     score_single_t.masked_fill_(R_mask, -torch.inf)
    #     _, T_t_indices = torch.topk(score_single_t, k=min(k, batch_size - 1 - k), dim=1)
    #
    #     # 5. 返回结果
    #     # 返回的是在当前 batch 内的相对索引
    #     return {
    #         'R': R_indices,
    #         'T_v': T_v_indices,
    #         'T_t': T_t_indices
    #     }

    def _find_graded_samples(self, batch_items, item_image_g, item_text_g, item_image_s, item_text_s, k=10):
        batch_size = len(batch_items)
        if batch_size < 3:
            # 样本太少，直接返回空索引（避免崩溃）
            empty = torch.empty((batch_size, 0), dtype=torch.long, device=batch_items.device)
            return {'R': empty, 'T_v': empty, 'T_t': empty}

        device = batch_items.device

        g_v = F.normalize(item_image_g[batch_items], dim=1)
        g_t = F.normalize(item_text_g[batch_items], dim=1)
        s_v = F.normalize(item_image_s[batch_items], dim=1)
        s_t = F.normalize(item_text_s[batch_items], dim=1)

        sim_g_v = g_v @ g_v.T
        sim_g_t = g_t @ g_t.T
        sim_s_v = s_v @ s_v.T
        sim_s_t = s_t @ s_t.T

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # 强正样本
        score_multi = sim_g_v * sim_g_t
        score_multi.masked_fill_(self_mask, -float('inf'))
        _, R_indices = torch.topk(score_multi, k=min(k, batch_size - 1), dim=1)

        # 排除已选强正 + 自身
        R_mask = torch.zeros_like(score_multi, dtype=torch.bool).scatter_(1, R_indices, True)
        R_mask.logical_or_(self_mask)

        # 弱正样本 - 视觉
        sim_s_v_norm = (sim_s_v + 1) / 2
        sim_s_t_norm = (sim_s_t + 1) / 2
        penalty_t = torch.clamp(1 - sim_s_t_norm, min=0.0)
        score_single_v = sim_s_v * penalty_t
        # 【核心修复】：使用原地的 masked_fill_ 替代 torch.where，完美避开数据类型冲突
        score_single_v.masked_fill_(score_single_v <= 0.1, -float('inf'))
        score_single_v.masked_fill_(R_mask, -float('inf'))

        t_k_v = max(1, min(k, batch_size - 1 - R_indices.shape[1]))  # 防止负数
        _, T_v_indices = torch.topk(score_single_v, k=t_k_v, dim=1)

        # 弱正样本 - 文本（同理）
        penalty_v = torch.clamp(1 - sim_s_v_norm, min=0.0)
        score_single_t = sim_s_t * penalty_v
        # 【核心修复】：同上
        score_single_t.masked_fill_(score_single_t <= 0.1, -float('inf'))
        score_single_t.masked_fill_(R_mask, -float('inf'))

        t_k_t = max(1, min(k, batch_size - 1 - R_indices.shape[1]))
        _, T_t_indices = torch.topk(score_single_t, k=t_k_t, dim=1)

        return {
            'R': R_indices,
            'T_v': T_v_indices,
            'T_t': T_t_indices
        }

    def _calculate_prototype_contrastive_loss(self, student_feats, graded_samples, batch_community_ids,
                                              sample_weights, modality_type: str, temp=0.2):
        """
        (新) 计算同时包含实例和原型对比的、并由样本难度加权的损失函数
        """
        student_feats = F.normalize(student_feats, dim=1)

        # ==========================================================
        # 【终极修复】：绝对不能在 batch 内现场算原型！
        # 必须查表获取 pre_epoch_processing 中算好的 "全局软锚点"
        # ==========================================================
        if modality_type == 'visual':
            global_protos = self.global_prototypes_v
        elif modality_type == 'textual':
            global_protos = self.global_prototypes_t
        else:
            raise ValueError("modality_type 必须是 'visual' 或 'textual'")

        # --- 2. 实例级和原型级相似度计算 ---
        sim_instance_matrix = student_feats @ student_feats.T / temp
        # 【修复】：使用全局原型矩阵计算相似度
        sim_proto_matrix = student_feats @ global_protos.T / temp

        # --- 3. 构造分子 (正样本) ---
        R_indices = graded_samples['R']
        if modality_type == 'visual':
            T_indices = graded_samples['T_v']
        else:
            T_indices = graded_samples['T_t']

        pos_instance_sim = torch.gather(torch.exp(sim_instance_matrix), 1, R_indices).sum(dim=1) + \
                           torch.gather(torch.exp(sim_instance_matrix), 1, T_indices).sum(dim=1)

        # 【修复】：直接根据物品的社区 ID 索引对应的全局原型
        proto_indices_per_sample = batch_community_ids.unsqueeze(1)
        pos_proto_sim = torch.gather(torch.exp(sim_proto_matrix), 1, proto_indices_per_sample).squeeze(1)

        numerator = pos_instance_sim + pos_proto_sim

        # --- 4. 构造分母 (所有负样本) ---
        denominator = torch.exp(sim_instance_matrix).sum(dim=1) + torch.exp(sim_proto_matrix).sum(dim=1)

        # --- 5. 计算加权损失 ---
        loss = -torch.log(numerator / (denominator + 1e-8))
        weighted_loss = (loss * sample_weights).mean()

        return weighted_loss

    def calculate_loss(self, interaction):
        """
        计算模型的总损失（所有损失逻辑合并后的版本）。
        """
        # 1. 前向传播，获取推荐分数并计算中间表示 (self.user_rep, self.result_embed_n1等)
        user, pos_items, neg_items = interaction
        pos_scores, neg_scores = self.forward(interaction)

        # =================================================================
        # 模块一: BPR损失计算
        # =================================================================
        # 使用最基础、标准的BPR损失
        loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # =================================================================
        # 模块二: 正则化损失计算
        # =================================================================
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        # =================================================================
        # 【关键修复】：提前提取当前批次的 valid items，供后续所有模块使用
        # =================================================================
        all_batch_items, _ = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        valid_mask = (all_batch_items >= 0) & (all_batch_items < self.n_items)
        all_batch_items = all_batch_items[valid_mask]

        # =================================================================
        # 模块三: 掩码/对比损失计算
        # =================================================================
        # 【修复】：原版这个 loss 是没有梯度的幽灵损失，激活它反而导致了特征坍塌。直接置 0。
        mask_f_loss = torch.tensor(0.0, device=self.device)

        # mask_g_loss: 图噪音对比损失 (恢复大规模负样本对比)
        # 【核心修复】：恢复全量用户的特征对比，提供极度丰富的负样本池。
        # 如果显存不够，请将其替换为随机采样 5000 个用户，但绝对不能局限于 batch 内的几百人！
        all_user_idx = torch.arange(self.n_users, device=self.device)
        mask_g_loss = self.InfoNCE(self.result_embed_n1[all_user_idx],
                                   self.result_embed_n2[all_user_idx],
                                   self.infoNCETemp)

        # 对于物品端，同样使用全局对比
        all_item_idx_shifted = torch.arange(self.n_items, device=self.device) + self.n_users
        mask_g_loss += self.InfoNCE(self.result_embed_n1[all_item_idx_shifted],
                                    self.result_embed_n2[all_item_idx_shifted],
                                    self.infoNCETemp)

        mask_g_loss = mask_g_loss * self.mask_weight_g

        loss_mask = mask_f_loss + mask_g_loss

        # =================================================================
        # 模块四: 引导式解耦损失计算
        # =================================================================

        loss_disentangle = torch.tensor(0.0, device=self.device, requires_grad=False)

        if all_batch_items.shape[0] <= 1:
            # 直接跳过解耦损失计算
            pass
        else:
            # a. 特征解耦
            item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
            student_g_img, student_g_txt = item_image_g[all_batch_items], item_text_g[all_batch_items]
            student_s_img, student_s_txt = item_image_s[all_batch_items], item_text_s[all_batch_items]

            # b. 自适应加权 (PCKD启发)
            with torch.no_grad():
                s_v_norm = F.normalize(student_s_img, dim=1)
                s_t_norm = F.normalize(student_s_txt, dim=1)
                difficulty_scores = 1 - (s_v_norm * s_t_norm).sum(dim=1)

                # 加温度避免 softmax 极端尖锐
                temperature = 0.1
                sample_weights = F.softmax(difficulty_scores / temperature, dim=0)

                # 归一化到合理范围，避免乘 batch_size 爆炸
                sample_weights = sample_weights / sample_weights.sum() * all_batch_items.shape[0]

            # c. 识别分级样本
            graded_samples = self._find_graded_samples(all_batch_items, item_image_g, item_text_g, item_image_s,
                                                       item_text_s, k=self.knn_k)
            # d. G特征对齐损失
            loss_InfoNCE_G = self.InfoNCE(student_g_img, student_g_txt, temp=self.infoNCETemp)

            # e. S特征分级原型对比损失
            batch_community_ids = self.item_community_map[all_batch_items]
            loss_S_visual = self._calculate_prototype_contrastive_loss(student_s_img, graded_samples,
                                                                       batch_community_ids, sample_weights,
                                                                       modality_type='visual', temp=self.infoNCETemp)
            loss_S_textual = self._calculate_prototype_contrastive_loss(student_s_txt, graded_samples,
                                                                        batch_community_ids, sample_weights,
                                                                        modality_type='textual', temp=self.infoNCETemp)
            loss_S_contrastive = loss_S_visual + loss_S_textual

            # f. 组合成加权的 graded loss
            loss_graded = self.alpha_g_s * loss_InfoNCE_G + (1 - self.alpha_g_s) * loss_S_contrastive

            # g. CLUB 独立性损失
            # 【核心修复】：必须对 student_g_img 加上 .detach()！
            # 保护通用语义空间不被互信息惩罚破坏，只让特有空间 S 承担解耦的梯度压力。
            raw_loss_club = self.item_image_estimator(student_s_img, student_g_img.detach()) + \
                            self.item_text_estimator(student_s_txt, student_g_txt.detach())

            loss_club = torch.clamp(raw_loss_club, min=0.0)

            # h. 最终的解耦损失
            loss_disentangle = self.lambda_1 * loss_club + self.lambda_graded_align * loss_graded

            # w/o GradedAlign  验证整个高级解耦与对比学习模块的整体优势。
            # loss_disentangle = self.lambda_1 * (loss_club + loss_InfoNCE_G)

        # =================================================================
        # 模块五: 加总所有损失
        # =================================================================
        total_loss = loss_bpr + reg_loss + loss_mask + loss_disentangle

        return total_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        # 【优化】使用向量化逻辑替代低效的逐个用户循环
        all_sampled_nodes = []
        all_sampled_weights = []

        for user_id in range(self.num_user):
            nodes, weights = self.user_graph_dict.get(user_id, ([], []))
            num_neighbors = len(nodes)

            if num_neighbors == 0:
                # 极端情况：无交互用户，填充 0 (【修复】：必须是 Tensor)
                all_sampled_nodes.append(torch.zeros(k, dtype=torch.long, device=self.device))
                all_sampled_weights.append(torch.zeros(k, dtype=torch.float, device=self.device))
                continue

            nodes_tensor = torch.tensor(nodes, device=self.device)
            weights_tensor = torch.tensor(weights, device=self.device)

            if num_neighbors >= k:
                sampled_nodes = nodes_tensor[:k]
                sampled_weights = weights_tensor[:k]
            else:
                # 邻居不足，利用 multinomial 或者 randint 进行带放回填充
                pad_indices = torch.randint(0, num_neighbors, (k - num_neighbors,), device=self.device)
                sampled_nodes = torch.cat([nodes_tensor, nodes_tensor[pad_indices]])
                sampled_weights = torch.cat([weights_tensor, weights_tensor[pad_indices]])

            all_sampled_nodes.append(sampled_nodes)
            all_sampled_weights.append(sampled_weights)

        # 堆叠为矩阵并一次性在 GPU 上计算 softmax
        nodes_matrix = torch.stack(all_sampled_nodes)
        weights_matrix = F.softmax(torch.stack(all_sampled_weights), dim=1)

        return nodes_matrix.cpu().tolist(), weights_matrix