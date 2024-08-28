# -*- coding: utf-8 -*-
# @Time : 2024/8/28 12:55
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 常用神经网络层.py

# 全连接层
import torch as t
import torch.nn as nn
# 输入 batch_size=2，维度3
input = t.randn(2, 3)
linear = nn.Linear(3, 4)
h = linear(input) #(2,4)
print(h)

print('\n'*5)

# DropOut层预防过拟合
# 每个元素以0.5的概率随机舍弃
import torch.nn as nn
dropout = nn.Dropout(0.5)
o = dropout(h)
print(o) # 有一半左右的数变为0,并对剩余元素进行缩放


print('\n'*5)


#  BatchNorm批标准化层
import torch.nn as nn
import torch as t
# 创建一个批标准化层，处理 4 个通道的输入（4个特征）
bn = nn.BatchNorm1d(4,0.01)
# 手动设置批标准化层的权重和偏置
bn.weight.data = t.ones(4) * 4   # 将权重初始化为 [4, 4, 4, 4]
bn.bias.data = t.zeros(4)        # 将偏置初始化为 [0, 0, 0, 0]
# 输入 o 的形状为 [batch_size, num_features]，其中 num_features=4
bn_out = bn(o)
# 输出经过批标准化后的张量
print(bn_out)
# 计算并打印输出张量每个特征的均值和方差
print(bn_out.mean(0))
print(bn_out.std(0, unbiased=False))























































