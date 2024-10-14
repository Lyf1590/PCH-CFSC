import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as cos
import random
import os
import settings
# import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity as cos
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(settings.GPU)
    torch.backends.cudnn.deterministic = True

def compress( database_loader, test_loader, model_gcn):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T,data_L, _) in enumerate(database_loader):
        var_data_I = Variable(data_I.cuda())
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        img_common, txt_common, code_I, code_T, _, _,_,_= model_gcn(var_data_I, var_data_T)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())
    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)


    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        # var_data_L = Variable(torch.FloatTensor(data_L.numpy()).cuda())
        img_common, txt_common, code_I, code_T, _, _,_,_= model_gcn(var_data_I,var_data_T)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def SimCLR(emb_i, emb_j, temperature=0.5):
    """
        计算SimCLR的对比损失。

        参数:
        - emb_i: 第一组嵌入 (batch_size, embedding_dim)
        - emb_j: 第二组嵌入 (batch_size, embedding_dim)
        - temperature: 温度参数，用于缩放相似度 (default=0.5)

        返回:
        - loss: 对比损失
        """
    batch_size = emb_i.shape[0]
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().cuda()
    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    positives = positives / temperature
    negatives = similarity_matrix / temperature

    nominator = torch.exp(positives)
    denominator = negatives_mask * torch.exp(negatives)  # 2*bs, 2*bs
    denominator = torch.sum(denominator, dim=1)

    loss_partial = -torch.log(nominator / denominator)  # 2*bs
    loss = torch.sum(loss_partial) / (2 * batch_size)

    return loss

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    # len = B2.shape[1]
    # distH = 0.5 * (len - np.dot(B1, B2.transpose()))
    # return distH
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calculate_hamming1(B1, B2):
    """
    Calculate Hamming distance between two sets of binary codes.
    :param B1: Binary codes for query samples
    :param B2: Binary codes for retrieval samples
    :return: Hamming distance matrix
    """
    q = B1.shape[0]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label 从x'z
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming1(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def p_topK2(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def zero2eps(x):
    x[x == 0] = 1
    return x


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty


 # 亲和度 多标签矩阵
def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    '''
    Use label or plabel to create the graph.
    :param tag1: img_label
    :param tag2: txt_label
    :return: adj
    '''
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)
    # 行，列
    return in_aff, out_aff, affinity_matrix


def p_recall(qu_B, re_B, qu_L, re_L):
    """
    计算完整的 Precision-Recall 曲线
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query labels
    :param re_L: {0,1}^{nxl} retrieval labels
    :return: 全部的 precisions, recalls, thresholds
    """
    num_query = qu_L.shape[0]
    all_precisions = []
    all_recalls = []

    for iter in tqdm(range(num_query)):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming1(qu_B[iter, :], re_B)
        precisions, recalls, _ = precision_recall_curve(gnd, -hamm)
        all_precisions.append(precisions)
        all_recalls.append(recalls)

    return all_precisions, all_recalls


def pr(all_precisions, all_recalls, recall_points):
    """
    在特定 recall 点插值 precision 值
    :param all_precisions: 所有查询的 precision 数组
    :param all_recalls: 所有查询的 recall 数组
    :param recall_points: 要计算的 recall 点
    :return: 每个 recall 点对应的平均 precision 值
    """
    interpolated_precisions = np.zeros(len(recall_points))

    for i, recall_point in enumerate(recall_points):
        precisions_at_recall = []
        for precisions, recalls in zip(all_precisions, all_recalls):
            if recall_point <= recalls.max():
                precisions_at_recall.append(np.interp(recall_point, recalls, precisions))
            else:
                precisions_at_recall.append(0.0)
        interpolated_precisions[i] = np.mean(precisions_at_recall)

    return interpolated_precisions


def pr_curve(qB, rB, qL, rL, recall_levels=None):
    """
    Input:
    qB: 查询集的二进制哈希码 [n_query, n_bits]
    rB: 检索集的二进制哈希码 [n_retrieve, n_bits]
    qL: 查询集的标签 [n_query, n_labels]
    rL: 检索集的标签 [n_retrieve, n_labels]
    recall_levels: 需要取的 recall 水平，默认取 [0, 0.2, 0.4, 0.6, 0.8, 1]
    """
    if recall_levels is None:
        recall_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    n_query = qB.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)

    P = []
    for recall_level in recall_levels:
        p = np.zeros(n_query)  # 各 query sample 的 Precision
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            hamm = calculate_hamming(torch.Tensor(qB[it, :]), torch.Tensor(rB)).squeeze()
            hamm = hamm.numpy()
            asc_id = np.argsort(hamm)

            if recall_level == 0:
                topk = 1
            else:
                topk = int(recall_level * gnd_all)
            if topk == 0:
                continue
            gnd = gnd[asc_id][:topk]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / topk

        P.append(np.mean(p))

    return P


def knn(num_node: int, k_num: int, feature1):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feature1.detach().cpu().numpy())
    col = np.argpartition(dist, -k_num, axis=1)[:, -k_num:]
    generated_features = torch.zeros((num_node, feature1.shape[1]))
    for i in range(num_node):
        similar_indices = col[i]
        similar_features = feature1[similar_indices]
        generated_feature = torch.mean(similar_features, dim=0)
        generated_features[i, :] = generated_feature

    return generated_features


def knn1(k_num: int, feature1):
    dist = cos(feature1.detach().cpu().numpy())
    col = np.argpartition(dist, -k_num, axis=1)[:, -k_num:]
    #nearest_features = feature1[col].mean(dim=1).cuda()
    nearest_features = torch.stack([feature1[idx].mean(dim=0) for idx in col]).cuda()
    return nearest_features

