# 目标：实现对图像的分类与识别

# 什么是卷积神经网络
## 卷积弯沉更多是对图像特征的提取或者说是信息匹配，当一个包含某些特征的图像经过一个卷积核的时候，
## 一些卷积核被激活，输出特定信号

# CNN架构
## 卷积层        conv2d
## 非线性变换层   tf.nn.relu/sigmiod/tanh
## 池化层        tf.nn.pool/tf.nn.avt, 主要目的是降采样，能改变形状
## 全连接层      w * x + b

# 卷积层的三个参数
## ksize   卷积核的大小
## strides 卷积核移动的跨度，可实现将采样， 一般都用1
## padding 边缘填充, 卷积核移动到边缘时，对外变的区域进行零填充，一般用SAME

import numpy as np
import tensorflow as tf



