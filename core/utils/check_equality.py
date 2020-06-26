import numpy as np


def check_equality(array1: np.ndarray, array2: np.ndarray, criterion=1E-6):
    array1 = array1 if isinstance(array1, np.ndarray) else np.array(array1)
    array2 = array2 if isinstance(array2, np.ndarray) else np.array(array2)
    assert array1.shape == array2.shape
    diff = array1 - array2
    diff = abs(diff)
    return sum(diff) / len(array1) < criterion
