import numpy as np

from core.poisson_2d import Poisson2D
from core.utils import check_equality


def test_five_point_zero():
    """测试完全为0的泊松方程"""
    calculator = Poisson2D(0.1, 0.1)
    assert check_equality(
        calculator.five_point_diff_1(
            np.zeros((4, 4)),
            *[np.ones((4, 3)) * [0, 1, 0]] * 4),
        np.zeros((2, 2)))


def test_five_point_constant():
    """测试常数泊松方程，这个方程的导数为0，边值全为4，结果也应全为4"""
    calculator = Poisson2D(0.1, 0.1)
    assert check_equality(calculator.five_point(np.zeros((2, 2)), *[np.ones(2) * 4] * 4, ), np.ones((2, 2)) * 4)


def test_five_point_constant_2():
    """测试常数泊松方程，这个方程的导数为0，边值全为4，结果也应全为4"""
    calculator = Poisson2D(0.1, 0.1)
    assert check_equality(calculator.five_point(np.zeros((4, 4)), *[np.ones(4) * 4] * 4, ), np.ones((4, 4)) * 4)


def test_five_point_x_diff():
    """u=x^2+1"""
    x, y = np.meshgrid(np.arange(0, 6, 1), np.arange(0, 6, 1))
    u = x * x * x + 1
    print(u)
    calculator = Poisson2D(1, 1)
    assert check_equality(calculator.five_point(
        delta_u=x[1:-1, 1:-1] * 6, top=u[0, 1:-1], bottom=u[-1, 1:-1], left=u[1:-1, 0], right=u[1:-1, -1],
    ), u[1:-1, 1:-1])


def test_five_point_2():
    """u=x+y"""
    x, y = np.meshgrid(np.arange(0, 6, 1), np.arange(0, 6, 1))
    u = x + y
    calculator = Poisson2D(1, 1)
    assert check_equality(calculator.five_point(
        delta_u=x[1:-1, 1:-1] * 0, top=u[0, 1:-1], bottom=u[-1, 1:-1], left=u[1:-1, 0], right=u[1:-1, -1],
    ), u[1:-1, 1:-1])


def test_five_point_3():
    """u=x+y+4"""
    x, y = np.meshgrid(np.arange(0, 6, 1), np.arange(0, 6, 1))
    u = x + y + 4
    calculator = Poisson2D(1, 1)
    assert check_equality(calculator.five_point(
        delta_u=x[1:-1, 1:-1] * 0, top=u[0, 1:-1], bottom=u[-1, 1:-1], left=u[1:-1, 0], right=u[1:-1, -1],
    ), u[1:-1, 1:-1])


def test_five_point_4():
    """u=y^2"""
    x, y = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 10, 1))
    u = y * y
    calculator = Poisson2D(1, 1)
    assert check_equality(calculator.five_point(
        delta_u=y[1:-1, 1:-1] * 0 + 2, top=u[0, 1:-1], bottom=u[-1, 1:-1], left=u[1:-1, 0], right=u[1:-1, -1],
    ), u[1:-1, 1:-1])
