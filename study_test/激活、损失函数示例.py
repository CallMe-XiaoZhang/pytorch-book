# -*- coding: utf-8 -*-
# @Time : 2024/8/28 14:13
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 激活、损失函数示例.py

# import torch.nn as nn
# import torch as t
#
# relu = nn.ReLU(inplace=True)
# input = t.randn(2, 3)
# print(input)
# output = relu(input)
# print(output) # 小于0的都被截断为0
# # 等价于input.clamp(min=0)


import torch.nn as nn
import torch as t
# batch_size=3，计算对应每个类别的分数（只有两个类别）
score = t.randn(3, 2) # 三个样本两个分类
print(score) # 模拟的经过人工神经网络得到的结果
# 三个样本分别属于1，0，1类，label必须是LongTensor
label = t.Tensor([1, 0, 1]).long()

# loss与普通的layer无差异
criterion = nn.CrossEntropyLoss(reduction='sum') # 实例化损失函数
loss = criterion(score, label)
print(loss)





























