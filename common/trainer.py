# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator

# 添加以下导入用于 t-SNE 可视化
import seaborn as sns
from sklearn.manifold import TSNE


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        # t-SNE 可视化相关配置
        self.tsne_enabled = config['tsne_enabled']
        self.tsne_interval = config['tsne_interval']  # 每隔多少个epoch生成一次t-SNE图
        self.tsne_sample_size = config['tsne_sample_size']  # 采样物品数量
        self.tsne_perplexity = config['tsne_perplexity']  # t-SNE的perplexity参数

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            
            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses
                    
                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()
                
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        # --- 新增: 定义模型保存路径和文件名 ---
        if saved:
            # 创建保存目录 (如果不存在)
            saved_dir = os.path.join(self.config['checkpoint_dir'], self.config['model'])
            os.makedirs(saved_dir, exist_ok=True)

            # 生成一个包含时间戳的唯一文件名
            file_name = f"{self.config['model']}-{self.config['dataset']}-{get_local_time()}.pth"
            saved_model_file = os.path.join(saved_dir, file_name)
        # --- 新增结束 ---

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # --- 新增: 生成 t-SNE 可视化 ---
            self._generate_tsne_visualization(epoch_idx)
            # --- 新增结束 ---

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                    # --- 新增: 在这里执行模型保存 ---
                    if saved:
                        torch.save(self.model.state_dict(), saved_model_file)
                        self.logger.info(f"██ Best model saved to: {saved_model_file}")
                    # --- 新增结束 ---

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def _generate_tsne_visualization(self, epoch_idx):
        """
        生成 t-SNE 可视化图，使用两种颜色区分文本模态和图像模态
        """
        if not self.tsne_enabled:
            return

        # 每隔指定的epoch才生成t-SNE图
        if epoch_idx % self.tsne_interval != 0:
            return

        self.logger.info(f'Generating t-SNE visualization for epoch {epoch_idx}...')

        try:
            # 将模型设置为评估模式
            self.model.eval()

            # 获取模型的物品嵌入
            with torch.no_grad():
                # 检查模型是否有必要的属性
                if hasattr(self.model, 'result_embed'):
                    all_embeddings = self.model.result_embed
                    n_users = self.model.n_users if hasattr(self.model, 'n_users') else 0

                    # 提取物品嵌入
                    item_embeddings = all_embeddings[n_users:]

                    # 采样物品
                    n_items = item_embeddings.shape[0]
                    sample_size = min(self.tsne_sample_size, n_items)
                    indices = np.random.choice(n_items, sample_size, replace=False)

                    sampled_embeddings = item_embeddings[indices].cpu().numpy()

                    # 创建模态标签 - 根据模型是否有视觉和文本特征来区分
                    modality_labels = []
                    visual_indices = []
                    textual_indices = []

                    # 检查模型是否有视觉特征和文本特征
                    has_visual = hasattr(self.model, 'v_feat') and self.model.v_feat is not None
                    has_textual = hasattr(self.model, 't_feat') and self.model.t_feat is not None

                    if has_visual and has_textual:
                        # 如果同时有视觉和文本特征，可以按某种方式区分
                        # 这里简单地将前半部分标记为视觉，后半部分标记为文本
                        mid_point = sample_size // 2
                        modality_labels = ['Visual'] * mid_point + ['Textual'] * (sample_size - mid_point)
                        visual_indices = list(range(mid_point))
                        textual_indices = list(range(mid_point, sample_size))
                    elif has_visual:
                        # 只有视觉特征
                        modality_labels = ['Visual'] * sample_size
                        visual_indices = list(range(sample_size))
                    elif has_textual:
                        # 只有文本特征
                        modality_labels = ['Textual'] * sample_size
                        textual_indices = list(range(sample_size))
                    else:
                        # 默认情况，简单分为两半
                        mid_point = sample_size // 2
                        modality_labels = ['Visual'] * mid_point + ['Textual'] * (sample_size - mid_point)
                        visual_indices = list(range(mid_point))
                        textual_indices = list(range(mid_point, sample_size))

                    # 计算视觉模态内部的平均距离
                    dist_v = 0.0
                    if len(visual_indices) > 1:  # 需要至少两个点才能计算距离
                        visual_embeddings = sampled_embeddings[visual_indices]
                        distances_v = []
                        for i in range(len(visual_embeddings)):
                            for j in range(i + 1, len(visual_embeddings)):
                                distance = np.linalg.norm(visual_embeddings[i] - visual_embeddings[j])
                                distances_v.append(distance)
                        dist_v = np.mean(distances_v) if distances_v else 0.0

                    # 计算文本模态内部的平均距离
                    dist_t = 0.0
                    if len(textual_indices) > 1:  # 需要至少两个点才能计算距离
                        textual_embeddings = sampled_embeddings[textual_indices]
                        distances_t = []
                        for i in range(len(textual_embeddings)):
                            for j in range(i + 1, len(textual_embeddings)):
                                distance = np.linalg.norm(textual_embeddings[i] - textual_embeddings[j])
                                distances_t.append(distance)
                        dist_t = np.mean(distances_t) if distances_t else 0.0

                    # 计算聚类质量评估指标
                    silhouette_score_val = 0.0
                    calinski_harabasz_score_val = 0.0
                    davies_bouldin_score_val = 0.0

                    try:
                        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

                        # 确保有足够的样本和类别来计算指标
                        unique_labels = list(set(modality_labels))
                        if len(unique_labels) > 1 and sample_size >= 2:
                            # 轮廓系数 (Silhouette Score)
                            label_numbers = [0 if label == 'Visual' else 1 for label in modality_labels]
                            silhouette_score_val = silhouette_score(sampled_embeddings, label_numbers)

                            # Calinski-Harabasz 指数
                            calinski_harabasz_score_val = calinski_harabasz_score(sampled_embeddings, label_numbers)

                            # Davies-Bouldin 指数
                            davies_bouldin_score_val = davies_bouldin_score(sampled_embeddings, label_numbers)
                    except Exception as e:
                        self.logger.warning(f"Error calculating clustering metrics: {e}")

                    # 执行 t-SNE
                    tsne = TSNE(n_components=2, perplexity=self.tsne_perplexity, random_state=42, max_iter=1000)
                    embeddings_2d = tsne.fit_transform(sampled_embeddings)

                    # 生成并保存图像
                    plt.figure(figsize=(12, 10))
                    sns.set_style("whitegrid")

                    # 使用两种颜色区分模态
                    palette = {"Visual": "blue", "Textual": "red"}
                    plot = sns.scatterplot(
                        x=embeddings_2d[:, 0],
                        y=embeddings_2d[:, 1],
                        hue=modality_labels,
                        palette=palette,
                        legend='full', alpha=0.8, s=50
                    )

                    plt.title(
                        f't-SNE Visualization of Item Embeddings (Epoch {epoch_idx})\n'
                        f'Dist_V: {dist_v:.4f}, Dist_T: {dist_t:.4f}\n'
                        f'Silhouette: {silhouette_score_val:.4f}, CH: {calinski_harabasz_score_val:.2f}, DB: {davies_bouldin_score_val:.4f}',
                        fontsize=14)
                    plt.xlabel('t-SNE Dimension 1', fontsize=12)
                    plt.ylabel('t-SNE Dimension 2', fontsize=12)
                    plt.tight_layout()

                    # 保存图像，文件名包含所有评估指标信息
                    save_dir = os.path.join(
                        self.config['checkpoint_dir'],
                        self.config['model'],
                        f'tsne_epoch_{epoch_idx}_dist_v_{dist_v:.4f}_dist_t_{dist_t:.4f}_'
                        f'sil_{silhouette_score_val:.4f}_ch_{calinski_harabasz_score_val:.2f}_db_{davies_bouldin_score_val:.4f}'
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"tsne_visualization.png"
                    filepath = os.path.join(save_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()

                    self.logger.info(f't-SNE visualization saved to {filepath}')
                    self.logger.info(f'Visual modality average distance (dist_v): {dist_v:.4f}')
                    self.logger.info(f'Textual modality average distance (dist_t): {dist_t:.4f}')
                    self.logger.info(f'Silhouette Score: {silhouette_score_val:.4f}')
                    self.logger.info(f'Calinski-Harabasz Score: {calinski_harabasz_score_val:.2f}')
                    self.logger.info(f'Davies-Bouldin Score: {davies_bouldin_score_val:.4f}')
                else:
                    self.logger.warning("Model does not have result_embed attribute, skipping t-SNE visualization")

        except Exception as e:
            self.logger.error(f"Error generating t-SNE visualization: {e}")

