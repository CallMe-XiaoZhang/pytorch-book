# -*- coding: utf-8 -*-
# @Time : 2024/8/28 14:52
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 优化器.py

import torch as t
import torch.nn as nn
from torch.nn import Linear

# 定义单隐藏层多层感知机
class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Perceptron, self).__init__()
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)

# 实例化网络
net = Perceptron(3, 4, 1)  # 3输入，4隐藏，1输出

# 定义优化器
from torch import optim
optimizer = optim.SGD(params=net.parameters(), lr=1)  # 随机梯度下降
optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()

# 定义均方误差损失函数
criterion = nn.MSELoss()

# 生成输入数据和目标标签
input = t.randn(32, 3)  # 输入32个样本，3特征
target = t.randn(32, 1)  # 目标32个样本，1特征，与输出维度一致
# 前向传播
output = net(input)
# 计算损失
loss = criterion(output, target)
# 反向传播,得到参数的梯度
loss.backward()
# 执行优化，根据梯度更新参数
optimizer.step()


# 查看优化器配置
# 分为两组，一组为权重，一组为偏执
weight_params = [param for name, param in net.named_parameters() if name.endswith('.weight')]
bias_params = [param for name, param in net.named_parameters() if name.endswith('.bias')]

# 重新定义优化器，对参数的不同部分设置不同学习率
optimizer = optim.SGD([
    {'params': bias_params},
    {'params': weight_params, 'lr': 1e-2}],
    lr=1e-5)
print(optimizer)

# 为选定层设定不同学习率，选定第一层

# special_layers 是一个 ModuleList，它包含了 net.layer1，此时第一层被选中
special_layers = nn.ModuleList([net.layer1])

# special_layers.parameters() 返回 special_layers 中所有参数的迭代器。
# map(id, special_layers.parameters()) 会获取这些参数的内存地址（或标识符，ID）。
# special_layers_params 是一个包含 special_layers 中参数 ID 的列表。
special_layers_params = list(map(id, special_layers.parameters()))# 由ID可确认special_layers参数
# base_params包含网络中所有不在 special_layers_params 列表中的参数
base_params = [param for param in net.parameters() if id(param) not in special_layers_params]
optimizer = t.optim.SGD([
            {'params': base_params},
            {'params': special_layers.parameters(), 'lr': 0.01} # 选定参数学习率调整为0.01
        ], lr=0.001 )
print(optimizer)
