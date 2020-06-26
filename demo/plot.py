import numpy as np
from core import plot


def test_plot_line():
    plot.plot_line([1, 2, 3], [3, 2, 3], 'b-')
    figure = plot.plot_line([1, 2, 3], [1, 3, 4], 'r-')
    figure.legend(['曲线1', '曲线2'], loc=(0.8, 0.15))
    figure.show()


def test_plot_surface():
    figure = plot.plot_surface(np.array([
        [1, 1, 1],
        [9, 1, 6],
        [-4, 2, 8],
    ]))
    figure.show()
