# 这个模块用于记录一维抛物型方程的差分函数
import numpy as _np
from numpy import ndarray as _ndarray


class Parabolic1D(object):
    def __init__(self, a: float = None, time_step: float = None, grid_step: float = None):
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
        return _np.hstack([data[1:], right])

    @staticmethod
    def _minus(data: _np.ndarray, left: float):
        return _np.hstack([left, data[:-1]])

    def forward_difference(self, data: _np.ndarray, left: float, right: float):
        data_plus = self._plus(data, right)
        data_minus = self._minus(data, left)
        u_next = data + self.a * self.lmd * (data_plus - 2 * data + data_minus)
        return u_next

    def backward_difference(self, data: _np.ndarray, left_next: float, right_next: float):
        """
        向后差分格式的计算
        u_{n-1} = A * u_{n+1}
        :param data: 前一层的数据
        :param left_next: 后一层的左侧数据
        :param right_next: 后一层的右侧数据
        :return:
        """
        data = data.copy()

        # 计算系数矩阵的逆
        matrix = _np.diag(_np.ones(len(data)) * (2 * self.a_lmd + 1))
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
