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
from IPython import get_ipython

# #接下来是一个简单的函数的求导过程，便于理解
# def f(x):
#     return 3 * x ** 2 - 4 * x
# def numerical_lim(f, x, h):
#     return (f(x + h) - f(x)) / h
# h=0.1
# for i in range(5):
#     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
#     h*=0.1
#     #那么接下来我们就要对我们求出来的函数以及导数进行可视化，便于理解
#     #这一步我们要使用matplotlib库，matplotlib是一个非常强大的可视化库，能够绘制各种各样的图形
#     #要配置环境，我们需要定义以下几个函数

# def use_svg_display(): #@save
#     """使用svg格式在Jupyter中显示图形."""
#     try:
#         ip=get_ipython()
#         if ip is not None:
#             ip.run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#             #backend_inline = get_ipython().kernel.get_parent().shell       #注意！想要在jupyter中使用svg格式显示图形，需要在jupyter中运行这段代码，而不是在终端中运行
#             #backend_inline.set_matplotlib_formats('svg')    #这一步是在jupyter中使用svg格式显示图形，svg格式的图形比png格式的图形更清晰
#         #之所以将上面两行命令注释掉，是因为在某些环境下可能会报错，所以我们使用try-except来捕获异常，避免程序崩溃，并且与40行的代码重复了，所以注释掉了
#     except (ImportError, AttributeError):
#         # 本地py脚本，无jupyter，提高dpi替代
#         plt.rcParams['figure.dpi'] = 150
#     #use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像,注意，#@save是d2l包的一个装饰器，表示将函数保存到d2l包中
#     #以后无需定义就可以直接使用它们
# def set_figsize(figsize=(3.5, 2.5)): #@save
#     """设置matplotlib的图表大小."""
#     use_svg_display()
#     d2l.plt.rcParams['figure.figsize'] = figsize
#     #定义set_figsize函数来设置图表大小。 
#     #注意，这里可以直接使用d2l.plt，因为导入语句 from matplotlib import pyplot as plt已标记为保存到d2l包中。
# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend): #@save
#     """设置matplotlib的轴."""
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
    #set_axes函数设置matplotlib的轴的属性，包括x轴和y轴的标签、刻度、范围和图例。


#     #现在前期布置准备好了，我们就可以开始绘制函数图像了，下面我们来定义一个绘图函数plot
#     #@save
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
#          ylim=None, xscale='linear', yscale='linear',
#          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []

#     set_figsize(figsize)
#     axes = axes if axes else d2l.plt.gca()#这里是获取当前的坐标轴对象，如果没有传入axes参数，就使用默认的坐标轴对象d2l.plt.gca()。

#     # 如果X有一个轴，输出True
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
#                 and not hasattr(X[0], "__len__"))

#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#     #这一段代码是在定义一个绘图函数plot，用于绘制数据点。
#     # 它接受多个参数，包括X和Y数据、标签、图例、坐标轴范围、坐标轴刻度类型、线条格式、图表大小和坐标轴对象。
#     # 函数内部首先设置图表大小，然后根据输入数据的维度进行处理，确保X和Y的数据格式正确。
#     # 最后，使用matplotlib的plot方法绘制数据点，并调用set_axes函数设置坐标轴属性。
# x = np.arange(0, 3, 0.1)#这一步是生成一个从0到3（不包括3）的数组，步长为0.1，用于绘制函数图像的x轴数据。
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# #现在我们已经定义了绘图函数plot，并使用它绘制了函数f(x)和切线2x-3的图像。 
# #想让他显示出来，我们需要调用d2l.plt.show()函数来显示图像。
# d2l.plt.show()
#----------------------------------------------------------------

# #接下来学习一下自动微分
x=torch.arange(4.0) #创建一个包含4个元素的张量，元素为0到3的整数

x.requires_grad_(True) #设置x的requires_grad属性为True，表示需要计算梯度
# y=2*torch.dot(x,x) 

# y.backward()
# #print(x.grad) #打印x的梯度，即dy/dx
# #print(x.grad == 4 * x) #打印x的梯度是否等于4x

# #接下来计算x的另一个函数
# x.grad.zero_() #将x的梯度清零
# y=x.sum()
# y.backward()
# print(x.grad) #打印x的梯度，即dy/dx

#下一步，分离计算
#有的时候，对于一个大模型，我们可能只想计算其中一部分的梯度，而不想计算整个模型的梯度，这时候就可以使用detach()方法来分离计算。
#比如，对于一个骨干权重已经训练好的模型，我们可能只想计算最后一层的梯度，而不想计算前面几层的梯度，这时候就可以使用detach()方法来分离计算。
# #这样做的好处是可以只训练并更新我们需要的层的梯度，不会影响其他层的梯度。
# x.grad.zero_() #先将x的梯度清零
# y = x * x 
# u = y.detach() #在这里，我们使用detach()方法将y分离出来，得到一个新的张量u，它与y共享数据，但不再计算梯度，也就不会将y的梯度传递给x。
# z = u * x

# z.sum().backward()
# x.grad == u 

#-------------------------------------------------
#概率
from torch.distributions import multinomial

#以掷骰子为例子，我们想要抽取一个样本，只需要输入一个概率分布向量和样本数量即可。
#比如我们想要从一个6面骰子中抽取一个样本，概率分布向量为[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]，样本数量为1，那么我们可以这样写：
probs = torch.ones(6) / 6
# output=multinomial.Multinomial(1, probs).sample() #输出一个样本，样本为一个one-hot向量，表示抽取的样本是哪一面骰子
# print(output) #输出样本

# #接下来我们想要抽取多个样本，比如我们想要从一个6面骰子中抽取10个样本，概率分布向量为[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]，样本数量为10，那么我们可以这样写：
# output1=multinomial.Multinomial(10, probs).sample() #输出10个样本，样本为一个one-hot向量，表示抽取的样本是哪一面骰子
# print(output1) #输出样本

# #好，那么接下来我们就可以模拟掷骰子的过程了，比如我们想要模拟掷骰子1000次，那么我们可以这样写：
# output2=multinomial.Multinomial(1000, probs).sample() #输出1000个样本，样本为一个one-hot向量，表示抽取的样本是哪一面骰子
# print(output2) #输出样本
# #然后我们可以计算相对频率，看看是否接近理论概率1/6。比如我们想要计算每一面骰子出现的次数，那么我们可以这样写：
# # relative_freq = output2 / 1000 #计算相对频率
# # print(relative_freq) #输出相对频率

# #下面，我们尝试进行500组模拟，每组模拟掷骰子1000次，看看每一面骰子出现的次数是否接近理论概率1/6。我们可以这样写：
# counts = multinomial.Multinomial(10, probs).sample((500,)) #输出500组样本，每组样本为一个one-hot向量，表示抽取的样本是哪一面骰子
# cum_counts = counts.cumsum(dim=0) #计算累计次数，cumsum()函数是对张量的元素进行累加，dim=0表示按行累加
# estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True) #计算相对频率，cum_counts.sum(dim=1, keepdims=True)表示按行求和，keepdims=True表示保持维度不变

# d2l.set_figsize((6, 4.5))
# for i in range(6):
#     d2l.plt.plot(estimates[:, i].numpy(),
#                  label=("P(die=" + str(i + 1) + ")"))
# d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
# d2l.plt.gca().set_xlabel('Groups of experiments')
# d2l.plt.gca().set_ylabel('Estimated probability')
# d2l.plt.legend();