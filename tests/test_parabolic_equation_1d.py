from core import parabolic_1d as module
import numpy as np
from core.utils import check_equality


def test_forward_difference():
    data = np.array([1, 2, 3])
    left = 4
    right = 5
    a = 1
    time_step = 0.1
    grid_step = 1
    calculator = module.Parabolic1D(data, left, right, a, time_step, grid_step)
    result = calculator.forward_difference()
    assert check_equality(result, np.array([1.4, 2, 3.1]))
