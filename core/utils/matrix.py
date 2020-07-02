import numpy as _np


def create_tridiagonal_matrix(length, center, upper, lower):
    """
    创建三对角矩阵
    :param length: 三对角矩阵的长和宽
    :param center: 中心的值
    :param upper: 上面的值
    :param lower: 下面的值
    :return:
    """
    matrix = _np.diag(_np.ones(length) * center)
    matrix[:-1, 1:] += _np.diag(_np.ones(length - 1) * upper)
    matrix[1:, :-1] += _np.diag(_np.ones(length - 1) * lower)
    return matrix
