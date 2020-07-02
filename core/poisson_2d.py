# 这个模块用于求解二维泊松方程

from .solver import Solver as _Solver
import numpy as _np
from numpy import ndarray as _ndarray
from .utils.matrix import create_tridiagonal_matrix as _create_tridiagonal_matrix


class Poisson2D(_Solver):
    def __init__(self, delta_x, delta_y):
        """
        设置求解器的参数
        :param delta_x:
        :param delta_y:
        """
        self.delta_x = delta_x
        self.delta_y = delta_y

    def five_point(self, delta_u: _ndarray, top: _ndarray, right: _ndarray, bottom: _ndarray, left: _ndarray):
        r"""
        采用五点差分格式求解泊松方程
        五点差分格式不需要角点的值
        编号的顺序如下所示：
        /   1   2   3   4   5   6   7   \
        |   8   9   10  11  12  13  14  |
        |   15  16  17  18  19  20  21  |
        |   22  23  24  25  26  27  28  |
        |   29  30  31  32  33  34  35  |
        |   36  37  38  39  40  41  42  |
        \   43  44  45  46  47  48  49  /
        u_{row,column}对应的编号为 row * len(columns) + column
        :param delta_u: shape=(row_length, col_length)
        :param top: shape=(col_length, )
        :param right: shape=(row_length, )
        :param bottom: shape=(col_length, )
        :param left: shape=(row_length, )
        :return:
        """
        # 仅用于正方形网格
        assert self.delta_x == self.delta_y

        # 检查并推算计算区域的行和列
        assert delta_u.shape[0] == right.shape[0] == left.shape[0]
        assert delta_u.shape[1] == top.shape[0] == bottom.shape[0]
        row_length, col_length = delta_u.shape

        # 求左侧的系数矩阵
        matrix = _np.zeros((row_length * col_length, row_length * col_length))
        for row_index in range(row_length):
            for col_index in range(col_length):
                center = row_index * col_length + col_index
                matrix[center, center] = 4  # 中心点
                if not col_index == 0:
                    matrix[center, center - 1] = -1  # 左
                if not col_index == col_length - 1:
                    matrix[center, center + 1] = -1  # 右
                if not row_index == 0:
                    matrix[center, center - col_length] = -1
                if not row_index == row_length - 1:
                    matrix[center, center + col_length] = -1

        vector = - delta_u.copy().reshape((row_length * col_length, 1)) * self.delta_x ** 2
        # 处理每一行的数据，添加左右边值
        for row_index in range(row_length):
            vector[row_index * col_length + 0, 0] += left[row_index]
            vector[(row_index + 1) * col_length - 1, 0] += right[row_index]
        # 处理每一列的数据，添加上下边值
        vector[:col_length, 0] += top.reshape(col_length)
        vector[-col_length:, 0] += bottom.reshape(col_length)

        result = _np.dot(_np.linalg.inv(matrix), vector)
        result = result.reshape(row_length, col_length)
        return result

    def five_point_diff_1(self, delta_u: _ndarray, top: _ndarray, right: _ndarray, bottom: _ndarray, left: _ndarray):
        row_length, col_length = delta_u.shape
        matrix = _np.zeros((row_length * col_length, row_length * col_length))
        vector = - delta_u.copy().reshape((row_length * col_length, 1)) * self.delta_x ** 2

        for row_index in range(row_length):
            for col_index in range(col_length):
                center = row_index * col_length + col_index
                if row_index in (0, row_length - 1) and col_index in (0, col_length - 1):
                    matrix[center, center] = 1  # 角点直接设置为1，即不参与计算
                elif row_index in (0, row_length - 1) or col_index in (0, col_length - 1):
                    if row_index == 0:  # 上边界
                        alpha, beta, gamma = top[col_index, :]
                        next_index = center + col_length
                        next_next_index = center + col_length * 2
                    elif row_index == row_length - 1:  # 下边界
                        alpha, beta, gamma = bottom[col_index, :]
                        next_index = center - col_length
                        next_next_index = center - col_length * 2
                    elif col_index == 0:  # 左边界
                        alpha, beta, gamma = left[row_index, :]
                        next_index = center + 1
                        next_next_index = center + 2
                    elif col_index == col_length - 1:  # 右边界
                        alpha, beta, gamma = right[row_index, :]
                        next_index = center - 1
                        next_next_index = center - 2
                    else:
                        raise ValueError
                    matrix[center, center] = alpha / 2 / self.delta_x
                    matrix[center, next_index] = beta
                    matrix[center, next_next_index] = -alpha / 2 / self.delta_x
                    vector[center] = gamma
                else:
                    matrix[center, center] = 4  # 中心点
                    matrix[center, center - 1] = -1  # 左
                    matrix[center, center + 1] = -1  # 右
                    matrix[center, center - col_length] = -1  # 上
                    matrix[center, center + col_length] = -1  # 下

        result = _np.linalg.inv(matrix)
        result = _np.dot(result, vector)
        result = result.reshape((row_length, col_length))
        return result[1:-1, 1:-1]
