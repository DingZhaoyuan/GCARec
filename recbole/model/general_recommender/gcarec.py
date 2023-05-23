

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F


class GCARec(GeneralRecommender):
    r"""
    拓扑级别自增强 + 特征级别自增强
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(GCARec, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.embed_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        # self.type = config["type"]
        self.ssl_tau = config["ssl_tau"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.eps = config["eps"]

        self.prob_edge_1 = config["prob_edge_1"]
        self.prob_edge_2 = config["prob_edge_2"]

        self.prob_feature_1 =config["prob_feature_1"]
        self.prob_feature_2 =config["prob_feature_2"]
        # 0.1 2.0

        self.user_embedding = torch.nn.Embedding(self.n_users, self.embed_dim, device=self.device)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embed_dim, device=self.device)
        self.reg_loss = EmbLoss()
        self.node_deg = self.calculate_node_deg()
        self.train_graph = self.create_adjust_matrix(is_sub=False)
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node.
        将增强后的邻接矩阵转为tensor
        """
        self.sub_graph1 = self.create_adjust_matrix(is_sub=True)
        self.sub_graph2 = self.create_adjust_matrix(is_sub=True)

    def calculate_node_deg(self):

        user = self._user
        item = self._item

        matrix = sp.csr_matrix((np.ones_like(user), (user, item + self.n_users)),
                               shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)).flatten()
        node_deg = torch.from_numpy(D).to(torch.float32).to(self.device)

        return node_deg

    # 获取节点维度drop概率
    def feature_drop_weights_dense(self, x, node_c):
        x = x.abs()
        w = x.t() @ node_c  # @等价于torch.matmul(,)表示矩阵相乘；*等价于torch.mul(,)表示元素间相乘
        w = w.log()
        s = (w.max() - w) / (w.max() - w.min())
        # print("s:",s)
        # print("s.shape:",s.shape)
        return s

    # 获取边drop概率
    def degree_drop_weights(self, edge_index):
        deg = self.node_deg
        # print("deg:",deg)
        # print("deg.shape:",deg.shape)
        deg_col = (deg[edge_index[1]].to(torch.float32) + deg[edge_index[0]].to(torch.float32)) / 2  # 根据节点的度计算边的度
        # print("deg_col:",deg_col)
        # print("deg_col.shape",deg_col.shape)
        s_col = torch.log(deg_col)
        # print("s_col:",s_col)
        # print("s_col.shape:",s_col.shape)
        weights = (s_col - s_col.min()) / (s_col.max() - s_col.min())
        # print("weights:",weights)
        # print("weights.shape:", weights.shape)
        return weights

    # 根据概率丢边
    def drop_edge_weighted(self, edge_index, edge_weights, p1: float, p2:float):
        edge_weights = edge_weights * p1 + p2
        # print("edge_weights:",torch.sort(edge_weights))
        edge_weights = edge_weights.where(edge_weights <= 1, torch.ones_like(edge_weights) * 1)
        edge_weights = edge_weights.where(edge_weights >= 0, torch.ones_like(edge_weights) * 0)

        # print("edge_weights:", edge_weights)
        # print("edge_weights.shape:", edge_weights.shape)
        # edge_weights = torch.ones_like(edge_weights) * 0.9   #SGL drop ratio 0.1
        sel_mask = torch.bernoulli(edge_weights).to(torch.bool)  # 改动
        # print("sel_mask.shape:",sel_mask.shape)

        return edge_index[:, sel_mask]  # 只保留为True的索引

    # 根据概率掩蔽维度
    def drop_feature_weighted_2(self, x, w, p1: float, p2:float):
        w = w  * p1 + p2
        # print("w",torch.sort(w))
        w = w.where(w <= 1, torch.ones_like(w) * 1)
        w = w.where(w >= 0, torch.ones_like(w) * 0)

        drop_prob = w
        # print("1-drop_prob:", 1.-drop_prob)
        # print("drop_prob.shape:", drop_prob.shape)
        drop_mask = torch.bernoulli(1.0 - drop_prob).to(torch.bool)  # 要改动

        x = x.clone()
        x[:, drop_mask] = 0.  # 为True的索引被置为0

        return x

    def create_adjust_matrix(self, is_sub: bool):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.If it is a subgraph, it may be processed by
        node dropout or edge dropout.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            csr_matrix of the normalized interaction matrix.
        """
        matrix = None
        if not is_sub:
            ratings = np.ones_like(self._user, dtype=np.float32)
            matrix = sp.csr_matrix((ratings, (self._user, self._item + self.n_users)),
                                   shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        #生成增强的邻接矩阵
        else:
            edge_index = torch.stack([self._user, self._item + self.n_users], dim=0)
            # print("edge_index.sahpe:",edge_index.shape)
            #消融实验，均匀概率，伯努利中心性概率..
            drop_weights = self.degree_drop_weights(edge_index).to(self.device)
            edge = self.drop_edge_weighted(edge_index, drop_weights, p1=self.prob_edge_1, p2 = self.prob_edge_2) #0.7 0.8效果最好

            user = edge[0]
            item = edge[1]
            matrix = sp.csr_matrix((np.ones_like(user), (user, item)),
                                   shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return self.csr2tensor(D.dot(matrix).dot(D))

    def csr2tensor(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.

        Args:
            matrix (scipy.csr_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor([matrix.row.tolist(), matrix.col.tolist()]),
            torch.FloatTensor(matrix.data.astype(np.float32)), matrix.shape
        ).to(self.device)
        return x

    def forward(self, graph, is_sub=False):

        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = []

        # 轻量图卷积
        for i in range(self.n_layers):
            main_ego = torch.sparse.mm(graph, main_ego)
            if is_sub:

                random_noise = torch.rand_like(main_ego).cuda()
                # random_noise = torch.ones_like(main_ego).cuda()
                noise = torch.sign(main_ego) * F.normalize(random_noise, dim=-1) * self.eps

                # node_weights = self.node_drop_weights().to(self.device)
                # noise = self.add_noise_weights(noise, node_weights, p=self.drop_feature_rate_1, lower_limit=0, upper_limit=1)

                node_deg = self.node_deg
                feature_weights = self.feature_drop_weights_dense(main_ego, node_c=node_deg).to(self.device)
                noise = self.drop_feature_weighted_2(noise, feature_weights, p1 = self.prob_feature_1, p2 = self.prob_feature_2)

                main_ego += noise
            all_ego.append(main_ego)
        # 层组合操作
        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)

        user_emd, item_emd = torch.split(all_ego, [self.n_users, self.n_items], dim=0)

        return user_emd, item_emd

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        user_emd, item_emd = self.forward(self.train_graph)
        # 拓扑自适应增强
        user_sub1, item_sub1 = self.forward(self.sub_graph1, True)
        user_sub2, item_sub2 = self.forward(self.sub_graph2, True)
        total_loss = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list) + \
                     self.calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2)
        return total_loss

    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)  # 这里进行改进，改进前进行实验
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called.

        """

        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T

