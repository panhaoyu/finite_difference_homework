from core import parabolic_1d as module
import numpy as np
from core.utils import check_equality


def test_all_methods():
    """这里的误差会偏大一些"""
    for a, time_step, grid_step, u0 in (
            (1, 1, 1, np.arange(1, 9)),
            (4, 1, 1, np.arange(1, 9)),
            (10, 1, 1, np.arange(1, 9)),
            (4, 1, 1, np.arange(-200, 200) * np.arange(-200, 200)),
            (-4, 1, 1, np.arange(-200, 200) * np.arange(-200, 200)),
    ):
        calculator = module.Parabolic1D(a=a, time_step=time_step, grid_step=grid_step)
        u1 = calculator.forward_difference(u0[1:-1], u0[0], u0[-1])
        # 检测向后差分格式
        result = calculator.backward_difference(u0[2:-2], u1[0], u1[-1])
        assert check_equality(u1[1:-1], result, criterion=1E8)
