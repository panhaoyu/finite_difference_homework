import numpy as np
from matplotlib import pyplot as plt
import os
from . import settings

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (5.771, 5.771 * 0.618)  # 图片大小，单位是inches
np.set_printoptions(edgeitems=30, linewidth=200000)  # 输出不换行
np.set_printoptions(precision=2, suppress=True)  # 精度，不使用科学计数
np.set_printoptions(threshold=np.inf)  # 过多的行，不会变成省略号


def check_and_set_ticks(axes, xticks, yticks):
    xticks is not None and axes.xaxis.set_ticks(xticks)
    xticks is not None and axes.xaxis.limit_range_for_scale(min(xticks), max(xticks))
    yticks is not None and axes.yaxis.set_ticks(yticks)
    yticks is not None and axes.yaxis.limit_range_for_scale(min(yticks), max(yticks))


def plot_line(
        x, y, *args, figure: plt.Figure = None, axes: plt.Axes = None,
        xticks=None, yticks=None, **kwargs):
    figure: plt.Figure = plt.gcf() if figure is None else figure
    if not axes:
        axes = plt.gca()

    # 设置刻度
    check_and_set_ticks(axes, xticks, yticks)

    axes.plot(x, y, *args, **kwargs)
    return figure


def plot_surface(
        data, show_colorbar=True, extent=None, vmin=None, vmax=None, xticks=None, yticks=None,
        axes: plt.Axes = None, figure: plt.Figure = None):
    assert len(set(map(len, data))) == 1
    # 对数据进行倒序，以保证初值在最下边
    data = data[::-1]
    # 获取图像和坐标轴对象
    figure = plt.figure() if figure is None else figure
    axes = plt.axes() if axes is None else axes
    # 计算最大值和最小值
    vmin = vmin is None and np.min(np.min(data)) or vmin
    vmax = vmax is None and np.max(np.max(data)) or vmax
    # 设置刻度
    check_and_set_ticks(axes, xticks, yticks)
    # 计算图像范围
    if not extent:
        if xticks is not None and yticks is not None:
            extent = (min(xticks), max(xticks), min(yticks), max(yticks))
        else:
            extent = (-1, 1, -1, 1)
    # 绘制二维图
    image = axes.imshow(
        data, vmin=vmin, vmax=vmax, interpolation='nearest',
        cmap=plt.cm.jet, extent=extent, aspect='auto')
    # 显示彩虹条图例
    if show_colorbar:
        plt.colorbar(image)

    return figure


def figure_path(file_name):
    path = os.path.join(settings.FIGURE_PATH, file_name)
    dir_name = os.path.dirname(path)
    os.path.exists(dir_name) or os.makedirs(dir_name)
    return path
