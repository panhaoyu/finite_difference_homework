# 这个模块用于记录一维抛物型方程的差分函数
import numpy as _np


def _plus(data: _np.ndarray, right: float):
    return _np.hstack([data[1:], right])


def _minus(data: _np.ndarray, left: float):
    return _np.hstack([left, data[:-1]])


def forward_difference(
        data: _np.ndarray, left: float, right: float,
        a: float, time_step: float, grid_step: float):
    """向前差分格式"""
    u_plus_1 = _plus(data, right)
    u_minus_1 = _minus(data, left)
    lmd = time_step / grid_step / grid_step
    u_next = data + a * lmd * (u_plus_1 - 2 * data + u_minus_1)
    return u_next
