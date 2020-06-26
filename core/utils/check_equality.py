import numpy as np


def check_equality(array1: np.ndarray, array2: np.ndarray, criterion=1E-5):
    assert array1.shape == array2.shape
    diff = array1 - array2
    diff = abs(diff)
    return sum(diff) < criterion
