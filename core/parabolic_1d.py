# 这个模块用于记录一维抛物型方程的差分函数
import numpy as _np


class Parabolic1D(object):
    def __init__(self,
                 data: _np.ndarray, left: float, right: float,
                 a: float, time_step: float, grid_step: float,
                 ):
        self.data = data
        self.left = left
        self.right = right
        self.a = a
        self.time_step = time_step
        self.grid_step = grid_step
        self.lmd = self.time_step / self.grid_step / self.grid_step
        self.data_minus_1 = self._minus(self.data, self.left)
        self.data_plus_1 = self._plus(self.data, right)

    @staticmethod
    def _plus(data: _np.ndarray, right: float):
        return _np.hstack([data[1:], right])

    @staticmethod
    def _minus(data: _np.ndarray, left: float):
        return _np.hstack([left, data[:-1]])

    def forward_difference(self):
        u_next = self.data + self.a * self.lmd * \
                 (self.data_plus_1 - 2 * self.data + self.data_minus_1)
        return u_next

    def backward_difference(self):
        pass
