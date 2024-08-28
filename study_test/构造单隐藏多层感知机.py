# -*- coding: utf-8 -*-
# @Time : 2024/8/28 12:14
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 构造单隐藏多层感知机.py

import torch as t
from torch import nn

# 每层都是全连接层
class Linear(nn.Module):  # 继承nn.Module，必须重写构造函数（__init__)和前向传播函数（forward）
    # 重写构造函数
    def __init__(self, in_features, out_features):
        super().__init__()  # 等价于nn.Module.__init__(self)，常用super方式，表示继承nn.Module的初始化方法
        # nn.Parameter内的参数是网络可学习的参数
        self.w = nn.Parameter(t.randn(in_features, out_features))#形成了一个(in_features, out_features)形状矩阵
        self.b = nn.Parameter(t.randn(out_features))
    # 重写前向传播
    def forward(self, x):
        x = x.mm(self.w)  # 矩阵乘法，等价于x.@(self.w)要求x的列与w的行相等
        return x + self.b.expand_as(x) #此处x=xw

# 构造多层感知机
class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):#同样重写__init__函数与前向传播方法
        super(Perceptron, self).__init__()
        # 此处的Linear是前面自定义的全连接层
        self.layer1 = Linear(in_features, hidden_features) # 输入神经元与隐藏层之间是全连接
        self.layer2 = Linear(hidden_features, out_features)# 隐藏层与输出神经元之间是全连接
    def forward(self, x):
        x = self.layer1(x)# 先进行第一次全连接得到wx+b
        x = t.sigmoid(x)#   使用激活函数处理为sigmoid（wx+b）为第二次全连接输入
        return self.layer2(x) # 最后一次全连接为结果，无需再次激活

perceptron = Perceptron(3,4,1)
for name, param in perceptron.named_parameters():
    print(name, param.size())