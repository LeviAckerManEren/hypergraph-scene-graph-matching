import torch
from scipy.optimize import linear_sum_assignment

def sinkhorn_normalization(affinity_matrix, num_iters=20, eps=1e-8):

    M = affinity_matrix.clone()
    for _ in range(num_iters):
        # 行归一化
        row_sum = M.sum(dim=-1, keepdim=True) + eps
        M = M / row_sum
        # 列归一化
        col_sum = M.sum(dim=-2, keepdim=True) + eps
        M = M / col_sum
    return M

def hungarian_matching(soft_assignment):

    cost = -soft_assignment.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))
