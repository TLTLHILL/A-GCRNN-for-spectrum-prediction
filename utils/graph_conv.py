import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize


def calculate_laplacian_with_self_loop(matrix):
    # 将矩阵加上一个对角线矩阵（自环），对角线矩阵的值为 1
    matrix = matrix + torch.eye(matrix.size(0))
    # 计算矩阵 matrix 中每行的和
    row_sum = matrix.sum(1)
    # 对每个行和进行平方根取倒数运算，得到一个数组 d_inv_sqrt
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    # 如果某个元素的平方根取倒数为无穷大，则将其设为 0.0
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    # 生成对角线矩阵 d_mat_inv_sqrt，矩阵中的每个元素为 d_inv_sqrt 中的值
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # 将矩阵 matrix 与 d_mat_inv_sqrt 相乘，再将结果转置，再与 d_mat_inv_sqrt 相乘，得到标准化 Laplacian 矩阵
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian
