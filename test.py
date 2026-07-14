##绘制函数 y=f(x)=x^3-1/x 和其在点x=1处的切线的图像

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from matplotlib import pyplot as plt 
from dl_d2l import d2l_torch as d2l
from IPython import get_ipython
#构造函数f(x)=x^3-1/x
def f(x):
    return x ** 3 - 1 / x

#定义求导
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / (h)

#开始准备绘图的环境
def use_svg_display(): #@save
    """使用svg格式在Jupyter中显示图形."""
    try:
        ip=get_ipython()
        if ip is not None:
            ip.run_line_magic('config', "InlineBackend.figure_format = 'svg'")
    except (ImportError, AttributeError):
        plt.rcParams['figure.dpi'] = 150
def set_figsize(figsize=(3.5, 2.5)): #@save
    """设置matplotlib的图表大小."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend): #@save
    """设置matplotlib的轴."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点."""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif not hasattr(Y[0], "__len__"):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
x=np.arange(0, 5, 0.1)
plot(x, [f(x), numerical_lim(f, 1, 0.01) * (x - 1) + f(1)], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show()

