from core import plot
from matplotlib import pyplot as plt
import numpy as np

grid_step = 0.025
# 网格向外扩展4圈，用以计算第二类边界条件
x_ticks = np.arange(0 - grid_step * 2, 1 + grid_step * 3, grid_step)
y_ticks = np.arange(0 - grid_step * 2, 1 + grid_step * 3, grid_step)
x_matrix, y_matrix = np.meshgrid(x_ticks, y_ticks)

fig: plt.Figure = plt.figure(figsize=(4, 4))
axes: plt.Axes = fig.add_axes((0.1, 0.1, 0.8, 0.8))
axes.grid(which='both')
axes.set_xticks(x_ticks)
axes.set_yticks(y_ticks)
fig.savefig(plot.get_figure_path('q2_3/网格'))

row_length, col_length = x_matrix.shape
delta_u = x_matrix * 0 - 16

# 边界条件的三个值：α，β，g
boundary_left = np.ones((row_length, 3)) * [1, 0, 0]
boundary_right = np.ones((row_length, 3)) * [0, 1, 0]
boundary_top = np.ones((col_length, 3)) * [1, 1, 0]
boundary_bottom = np.ones((col_length, 3)) * [1, 0, 0]

# 求左侧的系数矩阵
matrix = np.zeros((row_length * col_length, row_length * col_length))
vector = - delta_u.copy().reshape((row_length * col_length, 1)) * grid_step ** 2

for row_index in range(row_length):
    for col_index in range(col_length):
        center = row_index * col_length + col_index
        if row_index in (0, row_length - 1) and col_index in (0, col_length - 1):
            matrix[center, center] = 1  # 角点直接设置为1，即不参与计算
        elif row_index in (0, row_length - 1) or col_index in (0, col_length - 1):
            if row_index == 0:  # 上边界
                alpha, beta, gamma = boundary_top[col_index, :]
                next_index = center + col_length
                next_next_index = center + col_length * 2
            elif row_index == row_length - 1:  # 下边界
                alpha, beta, gamma = boundary_bottom[col_index, :]
                next_index = center - col_length
                next_next_index = center - col_length * 2
            elif col_index == 0:  # 左边界
                alpha, beta, gamma = boundary_left[row_index, :]
                next_index = center + 1
                next_next_index = center + 2
            elif col_index == col_length - 1:  # 右边界
                alpha, beta, gamma = boundary_right[row_index, :]
                next_index = center - 1
                next_next_index = center - 2
            else:
                raise ValueError
            matrix[center, center] = alpha / 2 / grid_step
            matrix[center, next_index] = beta
            matrix[center, next_next_index] = -alpha / 2 / grid_step
            vector[center] = gamma
        else:
            matrix[center, center] = 4  # 中心点
            matrix[center, center - 1] = -1  # 左
            matrix[center, center + 1] = -1  # 右
            matrix[center, center - col_length] = -1  # 上
            matrix[center, center + col_length] = -1  # 下

result = np.linalg.inv(matrix)
result = np.dot(result, vector)
result = result.reshape((row_length, col_length))


def plot_and_save(data, name):
    delta_x = grid_step
    figure: plt.Figure = plt.figure()
    axes: plt.Axes = figure.add_axes((0.1, 0.1, 0.9, 0.8))
    plot.plot_surface(data, figure=figure, axes=axes, reverse=False,
                      extent=(0 - delta_x / 2, 1 + delta_x / 2, 0 - delta_x / 2, 1 + delta_x / 2))
    axes.set_xlim(0 - delta_x / 2, 1 + delta_x / 2)
    axes.set_ylim(0 - delta_x / 2, 1 + delta_x / 2)
    axes.set_xticks(np.arange(0, 1 + delta_x, 0.2))
    axes.set_yticks(np.arange(0, 1 + delta_x, 0.2))
    figure.savefig(plot.get_figure_path('q2_3/' + name))


plot_and_save(result[1:-1, 1:-1], '结果')
plot_and_save((result[1:-1, 2:-1] - result[1:-1, 1:-2]) / grid_step, '差分x')
plot_and_save((result[2:-1, 1:-1] - result[1:-2, 1:-1]) / grid_step, '差分y')
diff_x_2_plus_diff_y_2 = (- 4 * result[2:-2, 2:-2]
                          + result[1:-3, 2:-2] + result[3:-1, 2:-2]
                          + result[2:-2, 1:-3] + result[2:-2, 3:-1]) / grid_step ** 2
plot_and_save(diff_x_2_plus_diff_y_2, '二阶差分求和')
