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
    """一个较为复杂的系数，会存在误差"""
    calculator = module.Parabolic1D(a=4, time_step=0.01, grid_step=1)
    result = calculator.forward_difference(data=np.array([4, 9, 16]), left=1, right=25)
    assert check_equality(result, [4.08, 9.08, 16.08])
    result = calculator.backward_difference([4, 9, 16], left_next=1.08, right_next=25.08)
    assert check_equality(result, [4.08, 9.08, 16.08])
