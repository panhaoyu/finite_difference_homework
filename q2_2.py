from core import plot
from core import poisson_2d
from matplotlib import pyplot as plt
import numpy as np

# fig: plt.Figure = plt.figure(figsize=(3, 3))
# axes: plt.Axes = fig.add_axes((0.1, 0.1, 0.8, 0.8))
# axes.grid(which='both')
# axes.set_xticks(np.arange(0, 1.1, 0.1))
# axes.set_yticks(np.arange(0, 1.1, 0.1))
# fig.savefig(plot.get_figure_path('q2_2/网格'))

calculator = poisson_2d.Poisson2D(0.1, 0.1)
calculator.five_point(
    np.ones((3, 4), float) * 16,
    top=np.ones(4, float) * 0,
    bottom=np.ones(4, float) * 0,
    left=np.ones(3, float) * 1,
    right=np.ones(3, float) * 0,
)
