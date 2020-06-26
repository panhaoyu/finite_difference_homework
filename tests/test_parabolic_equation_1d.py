from core import parabolic_1d as module
import numpy as np
from core.utils import check_equality


def test_all_methods():
    """这里的误差会偏大一些"""
    for a, time_step, grid_step, u0 in (
            (1, 1, 1, np.arange(1, 15)),
            (4, 1, 1, np.arange(1, 15)),
            (10, 1, 1, np.arange(1, 15)),
            (10, 1, 1, np.arange(-100, 100)),
            (4, 1, 1, np.arange(-20, 20) * np.arange(-20, 20) / 1000),
            (-4, 1, 1, np.arange(-20, 20) * np.arange(-20, 20) / 1000),
            (4, 1, 1, np.sin(np.arange(-90, 90) / 180 * np.pi) / 1000),
    ):
        left, right = u0[1], u0[-2]
        calculator = module.Parabolic1D(a=a, time_step=time_step, grid_step=grid_step)
        u1 = calculator.forward_difference(u0[1:-1], u0[0], u0[-1])
        u2 = calculator.forward_difference(u1[1:-1], u1[0], u1[-1])
        left_next, right_next = u1[0], u1[-1]

        # 检测向后差分格式
        result = calculator.backward_difference(u0[2:-2], left_next, right_next)
        assert check_equality(u1[1:-1], result, criterion=1E-8)
        result = calculator.crank_nicolson(u0[2:-2], left, right, left_next, right_next)
        assert check_equality(u1[1:-1], result, criterion=1E-8)
        result = calculator.du_fort_frankel(u0[2:-2], u1[1:-1], left_next, right_next)
        assert check_equality(u2, result, criterion=1E-6)
