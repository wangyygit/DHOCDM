import argparse
import numpy as np
import random
import torch
import os
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
'''
def get_graph_kernel(A, coef=1):
    #根据邻接矩阵计算图卷积核
    dim = A.shape[0]
    L = A - np.identity(dim)
    D = np.diag(L.sum(axis=1))
    L = D - L
    L = L + coef * np.identity(dim)
    return L
'''



def time_split(T, step=30):
    # 对长序列进行截断以训练模型
    # T: [num_node, num_time, num_feature]
    start = 0
    end = step
    samples = []
    while end <= T.shape[1]:
        samples.append(T[:, start:end])
        start += 1
        end += 1
    return samples


def evaluate_result(causality_true, causality_pred, threshold):
    # max_row = causality_pred.max(axis=1)
    # causality_pred = causality_pred / (np.repeat(max_row[:, np.newaxis], max_row.shape[0], 1) + 1e-6)
    causality_pred[causality_pred > 1] = 1
    causality_true = np.abs(causality_true).flatten()
    causality_pred = np.abs(causality_pred).flatten()
    roc_auc = roc_auc_score(causality_true, causality_pred)
    fpr, tpr, _ = roc_curve(causality_true, causality_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(causality_true, causality_pred)
    pr_auc = auc(recall_curve, precision_curve)

    causality_pred[causality_pred > threshold] = 1
    causality_pred[causality_pred <= threshold] = 0
    precision, recall, F1, _ = prfs(causality_true, causality_pred)
    accuracy = accuracy_score(causality_true, causality_pred)

    evaluation = {'accuracy': accuracy, 'precision': precision[1], 'recall': recall[1], 'F1': F1[1],
                  'ROC_AUC': roc_auc, 'PR_AUC': pr_auc}
    plot = {'FPR': fpr, 'TPR': tpr, 'PC': precision_curve, 'RC': recall_curve}
    return evaluation, plot


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        acc: prediction right rate
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # accuracy
    acc = (B_true == B_est).mean()
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'acc': acc, 'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}
    # return [acc, fdr, tpr, fpr, shd, pred_size]


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass

def dense_actor_loss(reward, avg_baseline, predict_reward, log_softmax,
                     device=None) -> torch.Tensor:
    """Calculate actor loss for reward type is 'dense'"""

    #reward, avg_baseline, predict_reward, log_softmax = Validation.to_device(
    #    reward, avg_baseline, predict_reward, log_softmax, device=device
    #)
    predict_reward = predict_reward.detach() # [Batch size, 1]
    reward_baseline = reward - avg_baseline - predict_reward
    actor_loss = - torch.mean(reward_baseline * log_softmax, 0)

    return actor_loss


def dense_critic_loss(reward, avg_baseline, predict_reward,
                      device=None) -> torch.Tensor:
    """Calculate actor loss for reward type is 'dense'"""

    #reward, avg_baseline, predict_reward = Validation.to_device(
    #    reward, avg_baseline, predict_reward, device=device
    #)
    reward = reward.detach()
    critic_loss = F.mse_loss(reward - avg_baseline,  predict_reward)

    return critic_loss


def graph_prunned_by_coef(graph_batch, X, th=0.1):
    """
    for a given graph, pruning the edge according to edge weights;
    linear regression for each causal regression for edge weights and then thresholding
    :param graph_batch: graph
    :param X: dataset
    :return:
    """
    mse = []
    d = len(graph_batch)
    reg = LinearRegression(fit_intercept=False)
    W = []
    # X = np.array(X).T
    for i in range(d):
        col = np.abs(graph_batch[:, i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, :, -2].reshape(-1, d)[:, col]
        # mul = aux_matrix[:, i].reshape(1, -1)[:, col]
        # X_train = np.multiply(X_train, mul)
        y_train = X[:, :, -1].reshape(-1, d)[:, i]
        reg.fit(X_train, y_train)
        reg_coeff = reg.coef_
        y_pre = reg.predict(X_train)
        # print(y.shape)
        # print(y_pre.shape)
        mse.append(np.square(y_train - y_pre))
        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)

    return np.float32(np.abs(W) > th)


def graph_prunned_by_coef_2nd(graph_batch, X, th=0.3):
    """
    for a given graph, pruning the edge according to edge weights;
    quadratic regression for each causal regression for edge weights and then thresholding
    :param graph_batch: graph
    :param X: dataset
    :return:
    """
    d = len(graph_batch)
    reg = LinearRegression()
    poly = PolynomialFeatures()
    W = []

    for i in range(d):
        col = graph_batch[i] > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        X_train_expand = poly.fit_transform(X_train)[:, 1:]
        X_train_expand_names = poly.get_feature_names_out()[1:]

        y = X[:, i]
        reg.fit(X_train_expand, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )

        for ci in range(d):
            if col[ci]:
                xxi = 'x{}'.format(cj)
                for iii, xxx in enumerate(X_train_expand_names):
                    if xxi in xxx:
                        if np.abs(reg_coeff[iii]) > th:
                            new_reg_coeff[ci] = 1.0
                            break
                cj += 1
        W.append(new_reg_coeff)

    return W


def convert_graph_int_to_adj_mat(graph_int):
    # Convert graph int to binary adjacency matrix
    # TODO: Make this more readable
    return np.array([list(map(int, ((len(graph_int) - len(np.base_repr(curr_int))) * '0' + np.base_repr(curr_int))))
                     for curr_int in graph_int], dtype=int)