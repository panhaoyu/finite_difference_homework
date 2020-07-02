from core import plot
from core import poisson_2d
from matplotlib import pyplot as plt
import numpy as np

fig: plt.Figure = plt.figure(figsize=(4, 4))
axes: plt.Axes = fig.add_axes((0.1, 0.1, 0.8, 0.8))
axes.grid(which='both')
axes.set_xticks(np.arange(-1, 1.1, 0.1))
axes.set_xticklabels(['{:.1f}'.format(index / 10 - 1) if index % 2 == 0 else '' for index in range(22)])
axes.set_yticks(np.arange(-1, 1.1, 0.1))
fig.savefig(plot.get_figure_path('q2_2/网格'))


def draw_and_save(data, grid_step, title):
    delta_x = grid_step
    figure: plt.Figure = plt.figure()
    axes: plt.Axes = figure.add_axes((0.1, 0.1, 0.9, 0.8))
    plot.plot_surface(data, figure=figure, axes=axes,
                      extent=(-1 - delta_x / 2, 1 + delta_x / 2, -1 - delta_x / 2, 1 + delta_x / 2))
    axes.set_xlim(-1 - delta_x / 2, 1 + delta_x / 2)
    axes.set_ylim(-1 - delta_x / 2, 1 + delta_x / 2)
    axes.set_xticks(np.arange(-1, 1 + delta_x, 0.2))
    axes.set_yticks(np.arange(-1, 1 + delta_x, 0.2))
    figure.savefig(plot.get_figure_path('q2_2/' + title))


def calculate_and_draw(grid_step, title) -> np.ndarray:
    delta_x = delta_y = grid_step
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
    draw_and_save(data, grid_step, title)
    return data


data1 = calculate_and_draw(0.1, '结果')
data2 = calculate_and_draw(0.025, '结果-细分')
# data3 = calculate_and_draw(0.0125, '结果-更细分')
error2 = data1 - data2[::4, ::4]
# error3 = data2 - data3[::2, ::2]
draw_and_save(error2, 0.1, '误差')
# draw_and_save(error2, 0.0125, '误差2')
# 第三次运算的算力要求太大，不做
