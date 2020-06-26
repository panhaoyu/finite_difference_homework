import numpy as np


def check_equality(array1: np.ndarray, array2: np.ndarray, criterion=1E-10):
    """
    比较两个矩阵是否相等
    :param array1: 第一个矩阵
    :param array2: 第二个矩阵
    :param criterion: 两矩阵作差的标准差
    :return:
    """
    array1 = array1 if isinstance(array1, np.ndarray) else np.array(array1)
    array2 = array2 if isinstance(array2, np.ndarray) else np.array(array2)
    assert array1.shape == array2.shape
    diff = array1 - array2
    return np.sqrt(sum(diff * diff) / len(array1)) < criterion
