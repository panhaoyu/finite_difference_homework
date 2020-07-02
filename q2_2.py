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

delta_x = delta_y = 0.025
calculator = poisson_2d.Poisson2D(delta_x, delta_y)
x, y = np.meshgrid(np.arange(-1, 1 + delta_x, delta_x), np.arange(-1, 1 + delta_y, delta_y))
result = calculator.five_point(
    delta_u=x[1:-1, 1:-1] * 0 + 16,
    top=x[0, 1:-1] * 0,
    bottom=x[-1, 1:-1] * 0,
    left=x[1:-1, 0] * 0,
    right=x[1:-1, -1] * 0,
)
data = x * 0
data[1:-1, 1:-1] = result
figure: plt.Figure = plt.figure()
axes: plt.Axes = figure.add_axes((0.1, 0.1, 0.9, 0.8))
plot.plot_surface(data, figure=figure, axes=axes,
                  extent=(-1 - delta_x / 2, 1 + delta_x / 2, -1 - delta_x / 2, 1 + delta_x / 2))
axes.set_xlim(-1 - delta_x / 2, 1 + delta_x / 2)
axes.set_ylim(-1 - delta_x / 2, 1 + delta_x / 2)
axes.set_xticks(np.arange(-1, 1 + delta_x, 0.2))
axes.set_yticks(np.arange(-1, 1 + delta_x, 0.2))
figure.savefig(plot.get_figure_path('q2_2/结果-细分'))
