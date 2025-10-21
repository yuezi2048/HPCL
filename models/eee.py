# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

from utils.mi_estimator import CLUBSample


class EEE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(EEE, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
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
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.mask_weight_g = config['mask_weight_g']
        self.mask_weight_f = config['mask_weight_f']
        self.infoNCETemp = config['infoNCETemp']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.v_rep_gcn = None  # (新) 用于存储 GCN 的原始输出
        self.t_rep_gcn = None  # (新) 用于存储 GCN 的原始输出
        self.mm_adj = None

        self.mlp = nn.Linear(2 * dim_x, 2 * dim_x)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

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

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

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
            self.v_gcn_n1 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.v_feat)
            self.v_gcn_n2 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)
            self.t_gcn_n1 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.t_feat)
            self.t_gcn_n2 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        self.result_embed_n1 = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_n2 = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        # =================================================================
        # DGMRec 解耦模块 (简化版) START
        # =================================================================

        self.embedding_dim = dim_x

        self.pre_image_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)  # (修改) pre_
        self.pre_text_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)  # (修改) pre_
        self.pre_shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)  # (修改) pre_
        nn.init.xavier_uniform_(self.pre_image_encoder.weight)  # (修改) pre_
        nn.init.xavier_uniform_(self.pre_text_encoder.weight)  # (修改) pre_
        nn.init.xavier_uniform_(self.pre_shared_encoder.weight)  # (修改) pre_

        self.pre_image_encoder_s = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)  # (修改) pre_
        self.pre_text_encoder_s = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)  # (修改) pre_
        nn.init.xavier_uniform_(self.pre_image_encoder_s.weight)  # (修改) pre_
        nn.init.xavier_uniform_(self.pre_text_encoder_s.weight)  # (修改) pre_

        # GCN后 (Post) 编码器
        self.post_image_encoder = nn.Linear(self.dim_latent, self.embedding_dim).to(self.device)
        self.post_text_encoder = nn.Linear(self.dim_latent, self.embedding_dim).to(self.device)
        self.post_shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.post_image_encoder.weight)
        nn.init.xavier_uniform_(self.post_text_encoder.weight)
        nn.init.xavier_uniform_(self.post_shared_encoder.weight)

        self.post_image_encoder_s = nn.Linear(self.dim_latent, self.embedding_dim).to(self.device)
        self.post_text_encoder_s = nn.Linear(self.dim_latent, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.post_image_encoder_s.weight)
        nn.init.xavier_uniform_(self.post_text_encoder_s.weight)

        self.lambda_1 = config['lambda_1']
        self.lambda_2 = config['lambda_2']  # (新) 假设您在 config 中添加了 lambda_2
        self.lambda_3 = config['lambda_3']  # (新) 假设您在 config 中添加了 lambda_3

        self.act_g = nn.Tanh()

        self.init_mi_estimator()

        # =================================================================
        # DGMRec 解耦模块 (简化版) END
        # =================================================================

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
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

        # =================================================================
        # DGMRec MI 估计器训练 START
        # =================================================================

        # (修改) 任务 1: 获取 "Pre" 嵌入
        item_image_g_pre, item_text_g_pre, item_image_s_pre, item_text_s_pre = self.mge_pre()

        # (新) 任务 2: 临时运行 GCN 以获取 "Post" 嵌入
        with torch.no_grad():
            v_rep, _ = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
            t_rep, _ = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
            item_v_rep_post = v_rep[self.num_user:]
            item_t_rep_post = t_rep[self.num_user:]

        # (新) 获取 "Post" 嵌入
        item_image_g_post, item_text_g_post, item_image_s_post, item_text_s_post = self.mge_post(item_v_rep_post,
                                                                                                 item_t_rep_post)

        for _ in range(5):
            # (修改) 训练所有4个估计器
            self.pre_item_image_estimator.train();
            self.pre_item_text_estimator.train();
            self.post_item_image_estimator.train();  # (新)
            self.post_item_text_estimator.train();  # (新)

            item_rand_idx = torch.randperm(self.n_items)[:2048].to(self.device)

            loss_mi = 0.0
            # (修改) "Pre" 估计器损失
            loss_mi += self.pre_item_image_estimator.learning_loss(item_image_s_pre[item_rand_idx],
                                                                   item_image_g_pre[item_rand_idx])
            loss_mi += self.pre_item_text_estimator.learning_loss(item_text_s_pre[item_rand_idx],
                                                                  item_text_g_pre[item_rand_idx])

            # (新) "Post" 估计器损失
            loss_mi += self.post_item_image_estimator.learning_loss(item_image_s_post[item_rand_idx],
                                                                    item_image_g_post[item_rand_idx])
            loss_mi += self.post_item_text_estimator.learning_loss(item_text_s_post[item_rand_idx],
                                                                   item_text_g_post[item_rand_idx])

            self.optimizer_club.zero_grad()
            loss_mi.backward(retain_graph=True)
            self.optimizer_club.step()

        # (修改)
        self.pre_item_image_estimator.eval();
        self.pre_item_text_estimator.eval();
        self.post_item_image_estimator.eval();  # (新)
        self.post_item_text_estimator.eval();  # (新)

        # =================================================================
        # DGMRec MI 估计器训练 END
        # =================================================================

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
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
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def init_mi_estimator(self):
        # (修改) "Pre" 估计器
        self.pre_item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)
        self.pre_item_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)

        # (新) "Post" 估计器
        self.post_item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)
        self.post_item_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)

        # (修改) 将所有4个估计器的参数添加到优化器
        params = list(self.pre_item_image_estimator.parameters()) + \
                 list(self.pre_item_text_estimator.parameters()) + \
                 list(self.post_item_image_estimator.parameters()) + \
                 list(self.post_item_text_estimator.parameters())

        self.optimizer_club = torch.optim.Adam(params, lr=1e-4)

    def mge_pre(self):  # (修改) mge -> mge_pre
        # (修改) 使用 "pre_" 编码器
        item_image_g = F.sigmoid(
            self.pre_shared_encoder(self.act_g(self.pre_image_encoder(self.image_embedding.weight))))
        item_text_g = F.sigmoid(self.pre_shared_encoder(self.act_g(self.pre_text_encoder(self.text_embedding.weight))))

        item_image_s = F.sigmoid(self.pre_image_encoder_s(self.image_embedding.weight))
        item_text_s = F.sigmoid(self.pre_text_encoder_s(self.text_embedding.weight))
        return item_image_g, item_text_g, item_image_s, item_text_s

    def mge_post(self, v_rep_input, t_rep_input):
        # (新) GCN后 (Post) 嵌入生成
        # (v_rep_input 和 t_rep_input 应该是 [n_items, dim_latent] 形状)

        # 通用(General)特征 G'
        item_image_g_post = F.sigmoid(self.post_shared_encoder(self.act_g(self.post_image_encoder(v_rep_input))))
        item_text_g_post = F.sigmoid(self.post_shared_encoder(self.act_g(self.post_text_encoder(t_rep_input))))

        # 特定(Specific)特征 S'
        item_image_s_post = F.sigmoid(self.post_image_encoder_s(v_rep_input))
        item_text_s_post = F.sigmoid(self.post_text_encoder_s(t_rep_input))
        return item_image_g_post, item_text_g_post, item_image_s_post, item_text_s_post

    # =================================================================
    # DGMRec 辅助函数 (简化版) END
    # =================================================================

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index_dropt, self.edge_index, self.id_feat)

        # (新) 存储 GCN 原始输出 (在 unsqueeze 之前)
        self.v_rep_gcn = self.v_rep
        self.t_rep_gcn = self.t_rep

        self.v_rep_n1, _ = self.v_gcn_n1(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n1, _ = self.t_gcn_n1(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)
        self.v_rep_n2, _ = self.v_gcn_n2(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n2, _ = self.t_gcn_n2(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)

        representation = torch.cat((self.v_rep, self.t_rep), dim=1)

        representation_n1 = torch.cat((self.v_rep_n1, self.t_rep_n1), dim=1)
        representation_n2 = torch.cat((self.v_rep_n2, self.t_rep_n2), dim=1)

        self.v_rep = torch.unsqueeze(self.v_rep, 2)
        self.t_rep = torch.unsqueeze(self.t_rep, 2)
        self.id_rep = torch.unsqueeze(self.id_rep, 2)

        user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        self.v_rep_n1 = torch.unsqueeze(self.v_rep_n1, 2)
        self.t_rep_n1 = torch.unsqueeze(self.t_rep_n1, 2)
        user_rep_n1 = torch.cat((self.v_rep_n1[:self.num_user], self.t_rep_n1[:self.num_user]), dim=2)
        user_rep_n1 = self.weight_u.transpose(1, 2) * user_rep_n1
        user_rep_n1 = torch.cat((user_rep_n1[:, :, 0], user_rep_n1[:, :, 1]), dim=1)

        self.v_rep_n2 = torch.unsqueeze(self.v_rep_n2, 2)
        self.t_rep_n2 = torch.unsqueeze(self.t_rep_n2, 2)
        user_rep_n2 = torch.cat((self.v_rep_n2[:self.num_user], self.t_rep_n2[:self.num_user]), dim=2)
        user_rep_n2 = self.weight_u.transpose(1, 2) * user_rep_n2
        user_rep_n2 = torch.cat((user_rep_n2[:, :, 0], user_rep_n2[:, :, 1]), dim=1)

        item_rep = representation[self.num_user:]
        item_rep_n1 = representation_n1[self.num_user:]
        item_rep_n2 = representation_n2[self.num_user:]

        h = self.buildItemGraph(item_rep)
        h_n1 = self.buildItemGraph(item_rep_n1)
        h_n2 = self.buildItemGraph(item_rep_n2)

        user_rep = user_rep
        item_rep = item_rep + h

        item_rep_n1 = item_rep_n1 + h_n1
        item_rep_n2 = item_rep_n2 + h_n2

        self.user_rep = user_rep
        self.item_rep = item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        self.user_rep_n1 = user_rep_n1
        self.item_rep_n1 = item_rep_n1
        self.result_embed_n1 = torch.cat((user_rep_n1, item_rep_n1), dim=0)

        self.user_rep_n2 = user_rep_n2
        self.item_rep_n2 = item_rep_n2
        self.result_embed_n2 = torch.cat((user_rep_n2, item_rep_n2), dim=0)

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

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        with torch.no_grad():
            u_temp, i_temp = self.user_rep.clone(), self.item_rep.clone()
            u_temp2, i_temp2 = self.user_rep.clone(), self.item_rep.clone()
            u_temp.detach()
            i_temp.detach()
            u_temp2.detach()
            i_temp2.detach()
            u_temp2 = self.mlp(u_temp2)
            i_temp2 = self.mlp(i_temp2)
            u_temp = F.dropout(u_temp, self.dropout)
            i_temp = F.dropout(i_temp, self.dropout)
        mask_loss_u = 1 - F.cosine_similarity(u_temp, u_temp2).mean()
        mask_loss_i = 1 - F.cosine_similarity(i_temp, i_temp2).mean()
        mask_f_loss = self.mask_weight_f * (mask_loss_i + mask_loss_u)

        mask_g_loss = (self.InfoNCE(self.result_embed_n1[:self.n_users], self.result_embed_n2[:self.n_users],
                                    self.infoNCETemp)
                       + self.InfoNCE(self.result_embed_n1[self.n_users:], self.result_embed_n2[self.n_users:],
                                      self.infoNCETemp))

        mask_g_loss = mask_g_loss * self.mask_weight_g

        # =================================================================
        # DGM 解耦逻辑 (修改后) START
        # =================================================================

        all_batch_items, _ = torch.unique(torch.cat((interaction[1], interaction[2])), return_inverse=True,
                                          sorted=False)

        valid_mask = (all_batch_items >= 0) & (all_batch_items < self.n_items)
        all_batch_items = all_batch_items[valid_mask]

        loss_disentangle_pre = torch.tensor(0.0).to(self.device)
        loss_disentangle_post = torch.tensor(0.0).to(self.device)
        loss_consistency = torch.tensor(0.0).to(self.device)

        if all_batch_items.shape[0] > 0:
            # --- 任务 1: GCN 前解耦 (使用 "pre" 模块) ---
            item_image_g_pre, item_text_g_pre, item_image_s_pre, item_text_s_pre = self.mge_pre()

            loss_InfoNCE_G_pre = self.InfoNCE(item_image_g_pre[all_batch_items], item_text_g_pre[all_batch_items],
                                              temp=self.infoNCETemp)

            loss_club_pre = 0.0
            loss_club_pre += self.pre_item_image_estimator(item_image_s_pre[all_batch_items],
                                                           item_image_g_pre[all_batch_items].detach())
            loss_club_pre += self.pre_item_text_estimator(item_text_s_pre[all_batch_items],
                                                          item_text_g_pre[all_batch_items].detach())

            loss_disentangle_pre = self.lambda_1 * (loss_club_pre + loss_InfoNCE_G_pre)

            # --- 任务 2: (新) GCN 后解耦 ---
            # (使用 self.forward() 中存储的 GCN 输出)
            item_v_rep_post = self.v_rep_gcn[self.num_user:]
            item_t_rep_post = self.t_rep_gcn[self.num_user:]

            item_image_g_post, item_text_g_post, item_image_s_post, item_text_s_post = self.mge_post(item_v_rep_post,
                                                                                                     item_t_rep_post)

            loss_InfoNCE_G_post = self.InfoNCE(item_image_g_post[all_batch_items], item_text_g_post[all_batch_items],
                                               temp=self.infoNCETemp)

            loss_club_post = 0.0
            loss_club_post += self.post_item_image_estimator(item_image_s_post[all_batch_items],
                                                             item_image_g_post[all_batch_items].detach())
            loss_club_post += self.post_item_text_estimator(item_text_s_post[all_batch_items],
                                                            item_text_g_post[all_batch_items].detach())

            loss_disentangle_post = self.lambda_2 * (loss_club_post + loss_InfoNCE_G_post)

            # --- 任务 3: (新) "特定计算公式" - GCN 前后一致性损失 ---
            # 约束 GCN "前" 的通用特征 和 GCN "后" 的通用特征 保持一致
            loss_consistency_v = self.InfoNCE(item_image_g_pre[all_batch_items], item_image_g_post[all_batch_items],
                                              temp=self.infoNCETemp)
            loss_consistency_t = self.InfoNCE(item_text_g_pre[all_batch_items], item_text_g_post[all_batch_items],
                                              temp=self.infoNCETemp)
            loss_consistency = self.lambda_3 * (loss_consistency_v + loss_consistency_t)

        # =================================================================
        # DGM 解耦逻辑 END
        # =================================================================

        # (修改) 最终总损失
        total_loss = loss_value + reg_loss + \
                     + mask_f_loss + mask_g_loss + \
                     loss_disentangle_pre + loss_disentangle_post + loss_consistency
        return total_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
            user_graph_index.append(user_graph_sample)

        return user_graph_index, user_weight_matrix


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1

        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
