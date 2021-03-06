# 这个模块用于计算第二次作业第1题

import numpy as np
from core import plot
from core.parabolic_1d import Parabolic1D
import matplotlib.pyplot as plt

x = np.arange(0, 1.1, 0.1)
u0 = np.sin(x * np.pi)
u_forward_history = np.array([u0])
u_backward_history = np.array([u0])
u_crank_history = np.array([u0])
t = np.arange(0, 0.105, 0.005)
left = t * 0
right = t * 0

calculator = Parabolic1D(a=1, time_step=0.005, grid_step=0.1)

for index in range(1, len(t)):
    t_i, t_next_i = t[index - 1], t[index]
    left_i, right_i = left[index - 1], right[index - 1]
    left_next_i, right_next_i = left[index], right[index]
    u_next = calculator.forward(data=u_forward_history[-1, 1:-1], left=left_i, right=right_i)
    u_forward_history = np.vstack([u_forward_history, np.hstack([left_next_i, u_next, right_next_i])])
    u_next = calculator.backward(data=u_backward_history[-1, 1:-1], left_next=left_next_i, right_next=right_next_i)
    u_backward_history = np.vstack([u_backward_history, np.hstack([left_next_i, u_next, right_next_i])])
    u_next = calculator.crank_nicolson(data=u_crank_history[-1, 1:-1], left=left_i, right=right_i,
                                       left_next=left_next_i, right_next=right_next_i)
    u_crank_history = np.vstack([u_crank_history, np.hstack([left_next_i, u_next, right_next_i])])

# 由于Du fort-Frankel格式是三层格式，需要先使用显式格式求得第二层才能进行
u_du_fort_history = u_forward_history[:2, :].copy()
for index in range(2, len(t)):
    t_i, t_next_i = t[index - 1], t[index]
    left_i, right_i = left[index - 1], right[index - 1]
    left_next_i, right_next_i = left[index], right[index]
    u_next = calculator.du_fort_frankel(data_this=u_du_fort_history[-1, 1:-1], data_prev=u_du_fort_history[-2, 1:-1],
                                        left_this=left_i, right_this=right_i)
    u_du_fort_history = np.vstack([u_du_fort_history, np.hstack([left_next_i, u_next, right_next_i])])

# 计算精确解
u_accurate = np.exp(-np.pi * np.pi * 0.1) * np.sin(np.pi * x)


def plot_procedure_and_result(u_history, name):
    figure: plt.Figure = plt.figure()
    grid = plt.GridSpec(2, 2, figure=figure, wspace=0.3, hspace=0.5)

    # 绘制二维图
    ax1: plt.Axes = figure.add_subplot(grid[0:2, 0])
    ax1.set_title('计算过程')
    plot.plot_surface(u_history, xticks=x[::2], yticks=t[::10], axes=ax1, show_colorbar=True)
    ax1.set_xlabel('x', labelpad=-3)
    ax1.set_ylabel('t', labelpad=-1)

    # 绘制结果图
    ax2: plt.Axes = figure.add_subplot(grid[0, 1])
    ax2.set_title('计算结果')
    ax2.set_ylim(0, 0.4)
    plot.plot_line(x, u_accurate, 'k-', axes=ax2, xticks=x[::2])
    plot.plot_line(x, u_history[-1, :], 'rx', axes=ax2, xticks=x[::2])
    ax2.set_xlabel('x', labelpad=-3)
    ax2.set_ylabel('u', labelpad=-1)
    ax2.legend(('精确解', name))

    ax3: plt.Axes = figure.add_subplot(grid[1, 1])
    plot.plot_line(x, (u_history[-1, :] - u_accurate) * 1E3, 'bx', axes=ax3, xticks=x[::2])
    ax3.set_ylim(-10, 20)
    ax3.set_xlabel('x', labelpad=-3)
    ax3.set_ylabel('$\Delta y/10^{-3}$', labelpad=-1)
    ax3.set_title('误差')

    figure.savefig(plot.get_figure_path(f'q2_1/λ=0.5/{name}'))


plot_procedure_and_result(u_forward_history, '向前差分')
plot_procedure_and_result(u_backward_history, '向后差分')
plot_procedure_and_result(u_crank_history, 'Crank Nicolson')
plot_procedure_and_result(u_du_fort_history, 'Du Fort-frankel')
