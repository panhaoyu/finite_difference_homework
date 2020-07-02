import numpy as np

from core.utils import matrix


def test_create_tridiagonal_matrix():
    assert matrix.create_tridiagonal_matrix(1, 1, 1, 1) == np.array([[1]])
    assert np.alltrue(matrix.create_tridiagonal_matrix(2, 1, 2, 3) == np.array([
        [1, 2],
        [3, 1],
    ]))
    assert np.alltrue(matrix.create_tridiagonal_matrix(3, 4, 5, 6) == np.array([
        [4, 5, 0],
        [6, 4, 5],
        [0, 6, 4],
    ]))
