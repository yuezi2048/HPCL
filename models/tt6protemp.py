# coding: utf-8
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models.GCN import GCN
from common.abstract_recommender import GeneralRecommender
from utils.mi_estimator import CLUBSample

class TT6PROTEMP(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TT6PROTEMP, self).__init__(config, dataset)

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
        # self.construction = 'weighted_max'
        self.reg_weight = config['reg_weight']

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

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # 1. 必须先准备好 mm_adj (邻接矩阵)，因为社区脚本 process_hierarchical_community_pro.py 依赖它
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        nn.init.xavier_uniform_(self.image_trs.weight) if self.v_feat is not None else None
        nn.init.xavier_uniform_(self.text_trs.weight) if self.t_feat is not None else None

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            # 如果矩阵不存在，现场计算并保存，解决版本不兼容报错问题
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj;
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # 2. 自动检查社区文件，不存在则调用脚本生成
        community_map_path = os.path.join(dataset_path, 'item_to_hierarchical_community.pt')
        if not os.path.exists(community_map_path):
            import subprocess
            import sys
            print(f"\n[自动构建] 正在生成 {config['dataset']} 的社区标签...\n")
            try:
                # 确保使用当前环境的 python 运行脚本
                subprocess.run([sys.executable, "process_hierarchical_community_pro.py", "-d", config['dataset'],
                                "-p", config['data_path']],
                               check=True)
            except Exception as e:
                print(f"[自动构建] 失败: {e}")

        # 3. 加载生成的社区文件
        if os.path.exists(community_map_path):
            self.register_buffer('item_community_map', torch.load(community_map_path).long())
        else:
            self.register_buffer('item_community_map', torch.randint(0, 100, (self.n_items,)).long())

        # 4. 加载用户图（保留原逻辑）
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

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

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)

        # =================================================================
        # DGMRec 解耦模块 (简化版) START
        # =================================================================

        # DGMRec 使用其 'embedding_size' 作为核心维度，我们将其映射到 MENTOR 的 dim_x
        self.embedding_dim = dim_x

        # 1. DGMRec 编码器 (用于分解 G 和 S)
        # (使用 MENTOR 的 self.v_feat 和 self.t_feat 作为输入维度)
        self.image_encoder = nn.Linear(self.feat_embed_dim, self.embedding_dim).to(self.device)
        self.text_encoder = nn.Linear(self.feat_embed_dim, self.embedding_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # 特定(Specific)特征编码器
        self.image_encoder_s = nn.Linear(self.feat_embed_dim, self.embedding_dim).to(self.device)
        self.text_encoder_s = nn.Linear(self.feat_embed_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder_s.weight)
        nn.init.xavier_uniform_(self.text_encoder_s.weight)

        # 2. DGMRec 超参数
        self.lambda_1 = config['lambda_1']

        self.act_g = nn.Tanh()

        # 3. DGMRec MI 估计器 (CLUB)
        self.init_mi_estimator()  # 调用辅助函数

        # ==================== 全局原型（来自 TT4PRO） ====================
        num_communities = torch.max(self.item_community_map).item() + 1
        self.register_buffer('global_prototypes_v', torch.zeros(num_communities, self.feat_embed_dim))
        self.register_buffer('global_prototypes_t', torch.zeros(num_communities, self.feat_embed_dim))

        # =================================================================
        # DGMRec 解耦模块 (简化版) END
        # =================================================================

        # 极简的门控层
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim_x * 2, dim_x),
            nn.Sigmoid()  # 只用一层，输出 0~1 的权重
        ).to(self.device)

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

        # ==================== 全局原型更新（TT4PRO 优势） ====================
        with torch.no_grad():
            _, _, item_image_s, item_text_s = self.mge()
            num_com = self.global_prototypes_v.shape[0]
            for m_feat, proto in zip([item_image_s, item_text_s],
                                     [self.global_prototypes_v, self.global_prototypes_t]):
                sum_feat = torch.zeros(num_com, self.feat_embed_dim, device=self.device)
                sum_feat.index_add_(0, self.item_community_map, m_feat)
                counts = torch.bincount(self.item_community_map, minlength=num_com).unsqueeze(1).clamp(min=1)
                proto.copy_(F.normalize(sum_feat / counts, dim=1))

        # =================================================================
        # DGMRec MI 估计器训练 START
        # =================================================================
        # (我们必须先生成 G/S 嵌入来训练估计器)
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()  #

        # DGM 默认训练 5 次
        for _ in range(5):
            self.item_image_estimator.train();
            self.item_text_estimator.train()  #

            # (DGM 使用 2048，如果项目数较少，请调整此值)
            item_rand_idx = torch.randperm(self.n_items)[:2048].to(self.device)

            loss_mi = 0.0
            # 计算 CLUB 学习损失
            loss_mi += self.item_image_estimator.learning_loss(item_image_s[item_rand_idx], item_image_g[item_rand_idx])
            loss_mi += self.item_text_estimator.learning_loss(item_text_s[item_rand_idx], item_text_g[item_rand_idx])

            self.optimizer_club.zero_grad()
            loss_mi.backward(retain_graph=True)  # 必须保留图，因为 mge() 的输出稍后会用于主损失
            self.optimizer_club.step()

        self.item_image_estimator.eval();
        self.item_text_estimator.eval()  #


    # =================================================================
    # DGMRec MI 估计器训练 END
    # =================================================================

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

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
        """核心升级：真正使用投影层 + L2 norm（解决 t-SNE 链状）"""
        # 投影 + 归一化
        if self.v_feat is not None:
            v_proj = self.image_trs(self.image_embedding.weight)
            v_proj = F.normalize(v_proj, dim=-1)
        else:
            v_proj = torch.zeros(self.num_item, self.feat_embed_dim, device=self.device)

        if self.t_feat is not None:
            t_proj = self.text_trs(self.text_embedding.weight)
            t_proj = F.normalize(t_proj, dim=-1)
        else:
            t_proj = torch.zeros(self.num_item, self.feat_embed_dim, device=self.device)

        # 通用特征 G（共享编码器）
        item_image_g = torch.sigmoid(self.shared_encoder(self.act_g(self.image_encoder(v_proj))))
        item_text_g = torch.sigmoid(self.shared_encoder(self.act_g(self.text_encoder(t_proj))))

        # 特定特征 S（直接用投影后的）
        item_image_s = torch.sigmoid(self.image_encoder_s(v_proj))
        item_text_s = torch.sigmoid(self.text_encoder_s(t_proj))

        return item_image_g, item_text_g, item_image_s, item_text_s

    def _generate_representations(self, v_rep, t_rep):
        # 1. 极其安全的残差门控
        concat_rep = torch.cat((v_rep, t_rep), dim=1)
        gate = self.fusion_gate(concat_rep)

        # 核心：原始特征 + 门控加权特征 (保证信息绝对不丢失，尺度不缩水)
        v_enhanced = v_rep + gate * v_rep
        t_enhanced = t_rep + (1.0 - gate) * t_rep
        representation = torch.cat((v_enhanced, t_enhanced), dim=1)

        # 2. 生成用户表征 (保持你的原逻辑不变，只替换输入)
        v_rep_u = torch.unsqueeze(v_enhanced[:self.num_user], 2)
        t_rep_u = torch.unsqueeze(t_enhanced[:self.num_user], 2)
        user_rep = torch.cat((v_rep_u, t_rep_u), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        # 3. 物品表征
        item_rep = representation[self.num_user:]
        h = self.buildItemGraph(item_rep)
        item_rep = item_rep + h

        result_embed = torch.cat((user_rep, item_rep), dim=0)
        return result_embed, user_rep, item_rep

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]

        # 1. 生成所有 GCN 输出
        # 主路径
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)

        # 2. 调用辅助方法来生成最终嵌入
        # 主嵌入 (需要保存 user_rep 和 item_rep 用于 mask_f_loss)
        self.result_embed, self.user_rep, self.item_rep = self._generate_representations(self.v_rep, self.t_rep)

        # 3. 计算分数用于 BPR 损失
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def _find_graded_samples(self, batch_items, item_image_g, item_text_g, item_image_s, item_text_s, k=10):
        """
        为批次中的每个物品动态识别分级样本。
        - 强正样本 (R): 通用特征在两个模态上都相似。
        - 弱正样本 (T_v, T_t): 特定特征在一个模态上相似，但在另一个模态的通用空间上有差异。
        """
        # 1. 提取当前批次的物品特征并进行归一化
        batch_size = len(batch_items)
        device = batch_items.device

        g_v = F.normalize(item_image_g[batch_items], dim=1)
        g_t = F.normalize(item_text_g[batch_items], dim=1)
        s_v = F.normalize(item_image_s[batch_items], dim=1)
        s_t = F.normalize(item_text_s[batch_items], dim=1)

        # 2. 计算所有相似度矩阵
        sim_g_v = g_v @ g_v.T
        sim_g_t = g_t @ g_t.T
        sim_s_v = s_v @ s_v.T
        sim_s_t = s_t @ s_t.T

        # 创建一个对角线为-inf的掩码，防止物品将自己选为邻居
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # 3. 识别强正样本 (Multi-modal Similar, R)
        score_multi = sim_g_v * sim_g_t
        score_multi.masked_fill_(self_mask, -torch.inf)
        _, R_indices = torch.topk(score_multi, k=min(k, batch_size - 1), dim=1)

        # 4. 识别弱正样本 (Single-modal Similar, T_v, T_t)
        R_mask = torch.zeros_like(score_multi, dtype=torch.bool).scatter_(1, R_indices, True)
        R_mask.logical_or_(self_mask)

        # 🚀 修复1: 在这里明确计算 valid_g_mask (加入通用空间门槛过滤，确保核心类别一致)
        # 如果相似度阈值 0.5 过于严格，你可以将其下调至 0.3
        valid_g_mask = (sim_g_v > 0.5) & (sim_g_t > 0.5)

        # 视觉弱正样本 T_v
        # 逻辑: 视觉特定属性相似，但用文本通用空间(G)的差异来惩罚
        score_single_v = sim_s_v * (1.0 - torch.tanh(1.5 * sim_g_t))
        score_single_v.masked_fill_(R_mask | ~valid_g_mask, -torch.inf)
        _, T_v_indices = torch.topk(score_single_v, k=min(k, batch_size - 1 - k), dim=1)

        # 文本弱正样本 T_t
        score_single_t = sim_s_t * (1.0 - torch.tanh(1.5 * sim_g_v))
        score_single_t.masked_fill_(R_mask | ~valid_g_mask, -torch.inf)
        _, T_t_indices = torch.topk(score_single_t, k=min(k, batch_size - 1 - k), dim=1)

        return {
            'R': R_indices,
            'T_v': T_v_indices,
            'T_t': T_t_indices
        }

    def _calculate_prototype_contrastive_loss(self, student_feats_current, student_feats_other, graded_samples,
                                              batch_community_ids, modality_type: str, temp=0.2):
        """TT6 干净版 + 同构联合概率空间 + 跨模态难度加权"""
        # 🚀 修复2: 区分当前模态 (current) 和另一模态 (other) 的特征输入
        student_feats_current = F.normalize(student_feats_current, dim=1)

        # 全局原型
        protos = self.global_prototypes_v if modality_type == 'visual' else self.global_prototypes_t

        sim_instance = student_feats_current @ student_feats_current.T / temp
        sim_proto = student_feats_current @ protos.T / temp

        # 难度加权（使用 current 和 other 计算真正的跨模态一致性偏差）
        with torch.no_grad():
            cross_modal_sim = (student_feats_current * F.normalize(student_feats_other, dim=1)).sum(dim=1)
            difficulty = 1.0 - cross_modal_sim
            weights = torch.sigmoid(0.5 * difficulty) + 0.5   # 安全区间 [0.5, 1.5]

        # 正样本（R + T）
        R_idx = graded_samples['R']
        T_idx = graded_samples['T_v'] if modality_type == 'visual' else graded_samples['T_t']
        pos_sim = torch.gather(torch.exp(sim_instance), 1, R_idx).sum(1) + \
                  torch.gather(torch.exp(sim_instance), 1, T_idx).sum(1)
        pos_proto = torch.gather(torch.exp(sim_proto), 1,
                                 batch_community_ids.unsqueeze(1)).squeeze(1)

        # 同构联合概率空间损失 (分母共享)
        loss = -torch.log((pos_sim + pos_proto) /
                          (torch.exp(sim_instance).sum(1) + torch.exp(sim_proto).sum(1) + 1e-8))
        return (loss * weights).mean()

    def calculate_loss(self, interaction):
        """
        计算模型的总损失（所有损失逻辑合并后的版本）。
        """
        # 1. 前向传播，获取推荐分数并计算中间表示 (self.user_rep)
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
        # 模块四: 引导式解耦损失计算
        # =================================================================
        loss_disentangle = torch.tensor(0.0, device=self.device)
        all_batch_items, _ = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        valid_mask = (all_batch_items >= 0) & (all_batch_items < self.n_items)
        all_batch_items = all_batch_items[valid_mask]

        if all_batch_items.shape[0] > 1:
            # a. 特征解耦
            item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
            student_g_img, student_g_txt = item_image_g[all_batch_items], item_text_g[all_batch_items]
            student_s_img, student_s_txt = item_image_s[all_batch_items], item_text_s[all_batch_items]

            # c. 识别分级样本
            graded_samples = self._find_graded_samples(all_batch_items, item_image_g, item_text_g, item_image_s,
                                                       item_text_s, k=self.knn_k)
            # d. G特征对齐损失
            loss_InfoNCE_G = self.InfoNCE(student_g_img, student_g_txt, temp=self.infoNCETemp)

            # e. S特征分级原型对比损失
            batch_community_ids = self.item_community_map[all_batch_items]

            # 🚀 修复2调用处: 交叉传入视觉和文本的 S 特征
            loss_S_visual = self._calculate_prototype_contrastive_loss(
                student_feats_current=student_s_img,  # 当前模态：视觉
                student_feats_other=student_s_txt,  # 另一模态：文本 (用于算跨模态难度)
                graded_samples=graded_samples,
                batch_community_ids=batch_community_ids,
                modality_type='visual',
                temp=self.infoNCETemp
            )

            loss_S_textual = self._calculate_prototype_contrastive_loss(
                student_feats_current=student_s_txt,  # 当前模态：文本
                student_feats_other=student_s_img,  # 另一模态：视觉
                graded_samples=graded_samples,
                batch_community_ids=batch_community_ids,
                modality_type='textual',
                temp=self.infoNCETemp
            )

            loss_S_contrastive = loss_S_visual + loss_S_textual

            # f. 组合成加权的 graded loss
            loss_graded = self.alpha_g_s * loss_InfoNCE_G + (1 - self.alpha_g_s) * loss_S_contrastive

            # g. CLUB 独立性损失
            loss_club = self.item_image_estimator(student_s_img, student_g_img.detach()) + \
                        self.item_text_estimator(student_s_txt, student_g_txt.detach())

            # h. 最终的解耦损失
            loss_disentangle = self.lambda_1 * loss_club + self.lambda_graded_align * loss_graded

            # w/o GradedAlign  验证整个高级解耦与对比学习模块的整体优势。
            # loss_disentangle = self.lambda_1 * (loss_club + loss_InfoNCE_G)

        # =================================================================
        # 模块五: 加总所有损失
        # =================================================================
        total_loss = loss_bpr + reg_loss + loss_disentangle

        return total_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        """
        (优化版) 为每个用户采样固定k个邻居及其权重。
        - 核心逻辑: 邻居数不足时，从已有邻居中随机采样进行填充。
        - 优化点: 代码更简洁，填充效率更高，数据类型处理更集中。
        """
        # 1. 初始化: 直接使用NumPy预分配内存，比Python列表更高效
        all_sampled_neighbors = np.zeros((self.num_user, k), dtype=np.int64)
        all_neighbor_weights = torch.zeros(self.num_user, k)

        # 2. 遍历所有用户进行采样
        for user_id in range(self.num_user):
            # 安全地获取邻居信息，如果用户没有邻居，则返回空列表
            neighbor_nodes = self.user_graph_dict.get(user_id, ([], []))[0]
            neighbor_weights = self.user_graph_dict.get(user_id, ([], []))[1]

            # 3. 统一处理邻居数量不足和过多的情况
            if len(neighbor_nodes) == 0:
                # 情况一：没有任何邻居，直接跳过，保留为0
                continue

            if len(neighbor_nodes) >= k:
                # 情况二：邻居数量充足，直接截取前k个
                sampled_nodes = neighbor_nodes[:k]
                sampled_weights = torch.tensor(neighbor_weights[:k])
            else:
                # 情况三：邻居数量不足，需要填充
                original_nodes = neighbor_nodes
                original_weights = torch.tensor(neighbor_weights)

                # 计算需要填充的数量
                num_to_pad = k - len(original_nodes)

                # 使用np.random.choice一次性高效地生成填充内容
                # replace=True表示可以重复采样
                padding_indices = np.random.choice(len(original_nodes), size=num_to_pad, replace=True)

                # 将原始邻居和填充邻居合并
                padded_nodes = [original_nodes[i] for i in padding_indices]
                padded_weights = original_weights[padding_indices]

                sampled_nodes = original_nodes + padded_nodes
                sampled_weights = torch.cat([original_weights, padded_weights])

            # 4. 将处理好的数据存入预分配的数组和张量中
            all_sampled_neighbors[user_id] = np.array(sampled_nodes)
            all_neighbor_weights[user_id] = F.softmax(sampled_weights, dim=0)  # 直接计算softmax

        # 5. 返回最终结果
        # all_sampled_neighbors可以直接使用，或者如果后续需要tensor，则转换为torch.tensor
        return all_sampled_neighbors.tolist(), all_neighbor_weights