# 这个模块用于记录一维抛物型方程的差分函数
import numpy as _np
from numpy import ndarray as _ndarray
from .solver import Solver as _Solver


class Parabolic1D(_Solver):
    def __init__(self, a: float, time_step: float, grid_step: float):
        """
        在对象进行初始化时，输入描述方程的一些参数，即在每一次时间步前进的过程中不变的数
        :param a: 方程中的常数
        :param time_step: 时间步长
        :param grid_step: 空间步长
        """
        self.a = a
        self.time_step = time_step
        self.grid_step = grid_step
        self.lmd = time_step / grid_step / grid_step
        self.a_lmd = a * self.lmd

    @staticmethod
    def _plus(data: _np.ndarray, right: float):
        data = data.copy()
        return _np.hstack([data[1:], right])

    @staticmethod
    def _minus(data: _np.ndarray, left: float):
        data = data.copy()
        return _np.hstack([left, data[:-1]])

    def forward(self, data: _np.ndarray, left: float, right: float):
        data = data.astype(float)
        data_plus = self._plus(data, right)
        data_minus = self._minus(data, left)
        u_next = data + self.a * self.lmd * (data_plus - 2 * data + data_minus)
        return u_next

    def backward(self, data: _np.ndarray, left_next: float, right_next: float):
        """
        向后差分格式的计算
        u_{n-1} = A * u_{n+1}
        :param data: 前一层的数据
        :param left_next: 后一层的左侧数据
        :param right_next: 后一层的右侧数据
        :return:
        """
        data = data.astype(float)

        # 计算系数矩阵的逆
        matrix = _np.diag(_np.ones(len(data)) * (2 * self.a_lmd + 1)).astype(float)
        for index in range(len(data) - 1):
            matrix[index, index + 1] -= self.a_lmd
            matrix[index + 1, index] -= self.a_lmd
        matrix = _np.linalg.inv(matrix)

        # 计算当前时间步的值，采用下一时间步的左右值进行修正
        data[0] += self.a_lmd * left_next
        data[-1] += self.a_lmd * right_next

        # 计算下一时间步的值
        data_next = _np.dot(matrix, data)
        return data_next

    def crank_nicolson(self, data, left, right, left_next, right_next):
        """
        采用Crank Nicolson进行计算

        :param data: 前一层的值
        :param left: 前一层的左边值
        :param right: 前一层的右边值
        :param left_next: 后一层的左边值
        :param right_next: 后一层的右边值
        :return:
        """
        b = self.a_lmd / 2  # 系数

        # 计算前一层的矩阵
        length = len(data)
        matrix_this = _np.diag(_np.ones(length) * (1 - 2 * b)).astype(float)
        matrix_this[1:, :-1] += _np.diag(_np.ones(length - 1) * b)
        matrix_this[:-1, 1:] += _np.diag(_np.ones(length - 1) * b)

        # 计算前一层的向量
        vector_this = _np.dot(matrix_this, data)
        vector_this = _np.transpose([vector_this])

        # 采用边值修正前一层的向量
        vector_this[0] += b * (left + left_next)
        vector_this[-1] += b * (right + right_next)

        # 计算下一层的矩阵
        matrix_next = _np.diag(_np.ones(length) * (1 + 2 * b)).astype(float)
        matrix_next[1:, :-1] += _np.diag(_np.ones(length - 1) * (-b))
        matrix_next[:-1, 1:] += _np.diag(_np.ones(length - 1) * (-b))

        u1 = _np.dot(_np.linalg.inv(matrix_next), vector_this)[0:, 0]
        return u1

    def du_fort_frankel(
            self, data_prev: _ndarray, data_this: _ndarray,
            left_this: float, right_this: float):
        """
        采用Du fort-Frankel格式进行计算
        :param data_prev: n-1层的数据
        :param data_this: n层的数据
        :param left_this: n层的左边值
        :param right_this: n层的右边值
        :return: n+1层的数据
        """
        b = 2 * self.a_lmd  # 方程的常数
        data_prev = data_prev.astype(float)
        data_this = data_this.astype(float)
        data_this_minus_1 = self._minus(data_this, left_this)
        data_this_plus_1 = self._plus(data_this, right_this)
        result = (data_prev * (1 - b) + b * (data_this_minus_1 + data_this_plus_1)) / (1 + b)
        return result
