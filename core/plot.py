import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (5.771, 5.771 * 0.618)  # 图片大小，单位是inches
np.set_printoptions(edgeitems=30, linewidth=200000)  # 输出不换行
np.set_printoptions(precision=2, suppress=True)  # 精度，不使用科学计数
np.set_printoptions(threshold=np.inf)  # 过多的行，不会变成省略号


def plot_line(x, y, *args, ax=None, **kwargs):
    figure: plt.Figure = plt.gcf()
    if not ax:
        ax = plt.gca()
    ax.plot(x, y, *args, **kwargs)
    return figure


def plot_surface(
        data, show_colorbar=True, extent=None, vmin=-10, vmax=10, xticks=None, yticks=None,
        axes: plt.Axes = None, figure: plt.Figure = None):
    assert len(set(map(len, data))) == 1
    # 对数据进行倒序，以保证初值在最下边
    data = data[::-1]
    # 获取图像和坐标轴对象
    figure = plt.figure() if figure is None else figure
    axes = plt.axes() if axes is None else axes
    # 设置刻度
    xticks and axes.xaxis.set_ticks(xticks)
    xticks and axes.xaxis.limit_range_for_scale(min(xticks), max(xticks))
    yticks and axes.yaxis.set_ticks(yticks)
    yticks and axes.yaxis.limit_range_for_scale(min(yticks), max(yticks))
    # 计算图像范围
    if not extent:
        if xticks and yticks:
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
