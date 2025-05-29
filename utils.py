import argparse
import numpy as np
import torch
import os
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs

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


def build_flags():
    parser = argparse.ArgumentParser()
    # 以下是全局参数
    parser.add_argument('--gpu-idx', type=int, default=0, help='set the gpu')
    parser.add_argument('--seed', type=int, default=620, help='Random seed.')
    parser.add_argument('--num-nodes', type=int, default=5,
                        help='Number of nodes in simulation.')
    parser.add_argument('--dims', type=int, default=1,
                        help='The number of input dimensions.')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='The threshold of evaluating causality.')
    parser.add_argument('--time-length', type=int, default=1000,
                        help='The length of time series.')

    # 以下是网络训练参数
    parser.add_argument('--val-epochs', type=int, default=300,
                        help='Number of epochs to train the val net.')
    parser.add_argument('--est-epochs', type=int, default=500,
                        help='Number of epochs to train the est net.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch.')
    parser.add_argument('--lr_val', type=float, default=1e-2,
                        help='Initial learning rate.')
    parser.add_argument('--lr_est', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay.')
    parser.add_argument('--sparsity-type', type=str, default='l2',
                        help='The type of sparsity loss.')
    parser.add_argument('--beta-sparsity', type=float, default=1,
                        help='The Weight of sparsity loss.')
    parser.add_argument('--beta-kl', type=float, default=1e-1,
                        help='The Weight of KL loss.')
    parser.add_argument('--beta-mmd', type=float, default=1,
                        help='The Weight of MMD loss.')

    # 以下是网络架构参数
    parser.add_argument('--est-hidden', type=int, default=15,
                        help='Number of hidden units.')
    parser.add_argument('--val-hidden', type=int, default=15,
                        help='Number of hidden units.')
    parser.add_argument('--est-dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--val-dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--root-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    return parser


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


def save_result(result, model_name, folder):
    filename = model_name + '_result.xls'
    filename = os.path.join(folder, filename)
    result = pd.DataFrame(result, index=[model_name])
    result.to_excel(filename)


def summary_result(data):
    idx = 0
    for temp in data:
        temp = pd.DataFrame(temp, index=[idx])
        if idx == 0:
            result = temp
        else:
            result = pd.concat([result, temp])
        idx += 1
    result = result.agg([np.mean, np.std])
    mean = result.loc['mean'].values
    std = result.loc['std'].values
    return mean, std



def kl_divergence(x, target):
    epsilon = 1e-6
    return torch.mean(x * torch.log((x+epsilon)/target))


def loss_sparsity(inputs, sparsity_type='l2', epsilon=1e-4):
    if sparsity_type == 'l1':
        return torch.mean(torch.abs(inputs))
    elif sparsity_type == 'log_sum':
        return torch.mean(torch.log(torch.abs(inputs) / epsilon + 1))
    else:
        return torch.mean(inputs ** 2)


def loss_divergence(inputs, divergence_type='entropy', rho=0.1):
    epsilon = 1e-6
    inputs = torch.abs(inputs)
    inputs = inputs.squeeze().mean(dim=2).mean(dim=0)
    if divergence_type == 'entropy':
        return -1 * torch.mean(inputs * torch.log(inputs + epsilon))
    elif divergence_type == 'JS':
        m = (rho + inputs) / 2
        return kl_divergence(inputs, m) / 2 + kl_divergence(rho, m) / 2
    else:
        return kl_divergence(inputs, rho)


def loss_mmd(x, y, idx, gamma=1):
    loss1 = torch.exp(-1 * gamma * (x - torch.repeat_interleave(x[:, idx:idx + 1, :, :], x.size(1), dim=1)) ** 2)
    loss2 = torch.exp(-1 * gamma * (x - torch.repeat_interleave(y.unsqueeze(1), x.size(1), dim=1)) ** 2)
    loss = torch.abs(torch.mean(loss1) - torch.mean(loss2))
    #loss = torch.mean((loss1 - loss2)*2)
    return loss

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