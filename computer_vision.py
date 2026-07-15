#在这个文件中来系统学习计算机视觉相关的操作
import torch
import torchvision
from torch import nn
from dl_d2l import d2l_torch as d2l
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path #在这里增加一个路径处理，在后续学习中会更加严谨。

#设置绘图大小
#plt.figure(figsize=(3.5, 3.5))
#这个设置可以手动设置，也可以不管他，在开启图像的时候会自己调的。


#一种简要快捷的写法：
img=Image.open('img/cat_piano.jpg') #这种写法目前是可以的，但是为了保证严谨，可以这么写：
#更加严谨不易出错的写法：
img_path=Path(__file__).parent/'img/cat_piano.jpg'
img=Image.open(img_path)

#除了查看图像以外，我们还要了解参数和图像信息
print("图像的类型：",img.mode)
print("图像的尺寸：",img.size)
##显示图片
# plt.imshow(img)
# plt.axis('off')
# plt.show()

#接下来我们定义一个apply函数来观察图像增广的效果
#这个函数的主要作用就是对同一张图像随机执行多次数据增强，然后拿出来做比较
#比如反转，随机裁剪，颜色抖动，旋转等
def apply(img, aug, num_rows=3, num_cols=3, scale=1.8): #aug是一个图像增广的函数，num_rows和num_cols是行数和列数，scale表示图像的缩放比例
    """定义一个apply函数来观察图像增广的效果"""
    Y = [aug(img) for _ in range(num_rows * num_cols)] #这是一个列表推导式，等价于：
    # Y=[]
    # for _ in range(num_rows * num_cols):
    #     result=aug(img)
    #     Y.append(result)
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    plt.show()

# apply(img, torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2)))
#来，这里面的randomresizedcrop是一个随机裁剪的函数
#里面的参数翻译成人话：从原图片中随机选择一个面积占原图10%～100%的区域，同时这个区域的宽高比例在0.5～2之间，然后把这个区域缩放成200×200
#也就是说这个只能用来裁剪，还有很多操作比如翻转，颜色抖动，旋转等，这些都是图像增广的操作，都是为了增加数据集的多样性，从而提高模型的泛化能力。
# #transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(),
#     transforms.ToTensor()
# ])
#--------------
#!!!!!在后续个人的项目中，比如说要做5分类，可以这么写(后面可以直接复制过去)：
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485,0.456,0.406],
#         std=[0.229,0.224,0.225]
#     )
# ])
#------
#接下来还有一个比较好玩的模块，就是学习颜色变化----
#在一些不需要通过颜色区别物体的训练学习中，可以改变同一物体的色调来增强学习
#考虑到识图对环境要求很严，同一物体在白天/晚上；室内/户外采集到的图像都会有不同
#所以我们需要尽可能多学习各种明度和色调的物体
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)
#也可以这么写：
#apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0))
#然后就是颜色"hue"(外星🐱要来了)
#apply(img,torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
#也可以调另两个：contrast对比度；saturation饱和度:
#apply(img,torchvision.transforms.ColorJitter(brightness=0.7,contrast=0.5,saturation=0.5,hue=0.5))
color_aug=torchvision.transforms.ColorJitter(brightness=0.7,contrast=0.5,saturation=0.5,hue=0.5)
#接下来尝试把刚才的几种图像处理方法结合起来：
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img,augs)
