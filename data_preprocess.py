import torch
import pandas as pd
import numpy as np

#v=torch.tensor([3.0,-4.0])
#print(torch.norm(v)) #L2 norm,也就是向量模长
#print(torch.norm(v,p=1)) #L1 norm,也就是向量各元素绝对值之和
##L1 norm 也有别的写法，在思路上就是求所有元素的绝对值之和，所以可以这样写：
#print(torch.abs(v).sum())
#print(torch.norm(v,p=float('inf'))) #L-infinity norm,也就是向量各元素绝对值的最大值

#同样，矩阵也有它对应的范数，名字叫做Frobenius norm，计算方式就是矩阵所有元素的平方和再开根号
#m_norm=torch.norm(torch.ones((4,9))) #Frobenius norm
#print(m_norm)

#--------------------
#微积分
#在深度学习中，训练模型变得更好就是最小化损失函数，最小化损失函数的方式就是求导数，求导数的方式就是微积分
from matplotlib import pyplot as plt
from dl_d2l import d2l_torch as d2l

#接下来是一个简单的函数的求导过程，便于理解
def f(x):
    return 3 * x ** 2 - 4 * x
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
h=0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h*=0.1
    #那么接下来我们就要对我们求出来的函数以及导数进行可视化，便于理解
    #这一步我们要使用matplotlib库，matplotlib是一个非常强大的可视化库，能够绘制各种各样的图形
    #要配置环境，我们需要定义以下几个函数
from IPython import get_ipython
def use_svg_display(): #@save
    """使用svg格式在Jupyter中显示图形."""
    try:
        ip=get_ipython()
        if ip is not None:
            ip.run_line_magic('config', "InlineBackend.figure_format = 'svg'")
            #backend_inline = get_ipython().kernel.get_parent().shell       #注意！想要在jupyter中使用svg格式显示图形，需要在jupyter中运行这段代码，而不是在终端中运行
            #backend_inline.set_matplotlib_formats('svg')    #这一步是在jupyter中使用svg格式显示图形，svg格式的图形比png格式的图形更清晰
        #之所以将上面两行命令注释掉，是因为在某些环境下可能会报错，所以我们使用try-except来捕获异常，避免程序崩溃，并且与40行的代码重复了，所以注释掉了
    except (ImportError, AttributeError):
        # 本地py脚本，无jupyter，提高dpi替代
        plt.rcParams['figure.dpi'] = 150
    #use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像,注意，#@save是d2l包的一个装饰器，表示将函数保存到d2l包中
    #以后无需定义就可以直接使用它们
def set_figsize(figsize=(3.5, 2.5)): #@save
    """设置matplotlib的图表大小."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
    #定义set_figsize函数来设置图表大小。 
    #注意，这里可以直接使用d2l.plt，因为导入语句 from matplotlib import pyplot as plt已标记为保存到d2l包中。
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
    #set_axes函数设置matplotlib的轴的属性，包括x轴和y轴的标签、刻度、范围和图例。


    #现在前期布置准备好了，我们就可以开始绘制函数图像了，下面我们来定义一个绘图函数plot
    #@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()#这里是获取当前的坐标轴对象，如果没有传入axes参数，就使用默认的坐标轴对象d2l.plt.gca()。

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
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
    #这一段代码是在定义一个绘图函数plot，用于绘制数据点。
    # 它接受多个参数，包括X和Y数据、标签、图例、坐标轴范围、坐标轴刻度类型、线条格式、图表大小和坐标轴对象。
    # 函数内部首先设置图表大小，然后根据输入数据的维度进行处理，确保X和Y的数据格式正确。
    # 最后，使用matplotlib的plot方法绘制数据点，并调用set_axes函数设置坐标轴属性。
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
#现在我们已经定义了绘图函数plot，并使用它绘制了函数f(x)和切线2x-3的图像。 
#想让他显示出来，我们需要调用d2l.plt.show()函数来显示图像。
d2l.plt.show()
