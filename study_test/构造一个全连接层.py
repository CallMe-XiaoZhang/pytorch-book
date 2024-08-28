# -*- coding: utf-8 -*-
# @Time : 2024/8/28 11:47
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 构造一个全连接层.py


# y=wx+b

import torch as t
from torch import nn
print(t.__version__)

# 此处实际构造的是y=xw+b
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

layer = Linear(4,3)
input = t.randn(2,4)# 两个样本，四个特征值
output = layer(input)# layer是构造的单层全连接的实例
print(output)

# - `Module`中的可学习参数可以通过`named_parameters()`或者`parameters()`返回一个迭代器，
#    前者会给每个parameter都附上名字，使其更具有辨识度。
for name, parameter in layer.named_parameters():
    print(name, parameter) # w and b



