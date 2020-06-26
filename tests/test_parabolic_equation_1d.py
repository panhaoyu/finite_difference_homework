from core import parabolic_1d as module
import numpy as np
from core.utils import check_equality


def test_accurate():
    """精确解，不存在误差"""
    accurate = np.array([2, 3, 4])
    calculator = module.Parabolic1D(a=1, time_step=1, grid_step=1)
    result = calculator.forward_difference(data=np.array([2, 3, 4]), left=1, right=5)
    assert check_equality(result, accurate, criterion=1E9)
    result = calculator.backward_difference(data=np.array([2, 3, 4]), left_next=1, right_next=5)
    assert check_equality(result, accurate, criterion=1E9)


def test_complex():
    """一个较为复杂的系数"""
    calculator = module.Parabolic1D(a=4, time_step=0.01, grid_step=1)
    result = calculator.forward_difference(data=np.array([4, 9, 16]), left=1, right=25)
    assert check_equality(result, [4.08, 9.08, 16.08])
    result = calculator.backward_difference([4, 9, 16], left_next=1.08, right_next=25.08)
    assert check_equality(result, [4.08, 9.08, 16.08])


def test_complex_2():
    """这里的误差会偏大一些"""
    calculator = module.Parabolic1D(a=1, time_step=0.01, grid_step=1)
    x = np.arange(1, 8)
    u0 = x
    u1 = calculator.forward_difference(data=u0[1:-1], left=u0[0], right=u0[-1])
    result = calculator.backward_difference(data=u0[2:-2], left_next=u1[0], right_next=u1[-1])
    assert check_equality(u1[1:-1], result, criterion=1E5)
