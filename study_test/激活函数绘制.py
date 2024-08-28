# -*- coding: utf-8 -*-
# @Time : 2024/8/28 14:04
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 激活函数绘制.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 创建输入数据
x = torch.linspace(-5, 5, 100)

# 实例化激活函数
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=0)
elu = nn.ELU(alpha=1.0)
selu = nn.SELU()
gelu = nn.GELU()
silu = nn.SiLU()
softplus = nn.Softplus(beta=1, threshold=20)

# 将输入数据扩展以便 softmax 使用
x_for_softmax = x.unsqueeze(0)

# 计算激活函数的输出
activations = {
    'ReLU': relu(x).detach().numpy(),
    'Leaky ReLU': leaky_relu(x).detach().numpy(),
    'Sigmoid': sigmoid(x).detach().numpy(),
    'Tanh': tanh(x).detach().numpy(),
    'Softmax': softmax(x_for_softmax).squeeze().detach().numpy(),
    'ELU': elu(x).detach().numpy(),
    'SELU': selu(x).detach().numpy(),
    'GELU': gelu(x).detach().numpy(),
    'Swish': silu(x).detach().numpy(),
    'Softplus': softplus(x).detach().numpy()
}

# 设置图像布局
fig, axes = plt.subplots(3, 4, figsize=(20, 12))

# 绘制每个激活函数的图像
for i, (name, output) in enumerate(activations.items()):
    row, col = divmod(i, 4)
    axes[row, col].plot(x.numpy(), output)
    axes[row, col].set_title(name)
    axes[row, col].grid(True)

# 隐藏第三行多余的子图
for j in range(2, 4):
    axes[2, j].axis('off')

# 调整布局
plt.tight_layout()
plt.show()