# -*- coding: utf-8 -*-
# @Time : 2024/8/19 17:46
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 用autograd实现线性回归.py

import torch as t
from matplotlib import pyplot as plt
import numpy as np
import MyFunction

# 设置随机数种子，保证结果可复现
MyFunction.same_seed(240819)

# 产生数据
def get_fake_data(batch_size=8):
    '''
    产生随机数据：y = x * 2 + 3，并加上一些噪声
    返回值:
    - x: 随机输入变量，范围在 [0, 5) 之间
    - y: 根据线性方程生成的输出，加上一些随机噪声
    '''
    x = t.rand(batch_size, 1) * 5  # 生成 [0, 5) 之间的随机数，大小为 batch_size x 1
    y = x * 2 + 3 + t.randn(batch_size, 1)  # y = 2 * x + 3 加噪声
    return x, y

# 来看看产生的 x-y 分布是什么样的
x, y = get_fake_data()
plt.scatter(x.detach().numpy(), y.detach().numpy())  # 绘制生成的数据点
plt.show()

# 随机初始化参数
w = t.rand(1, 1, requires_grad=True)  # 初始化 w 为一个随机数，并需要计算梯度
b = t.zeros(1, 1, requires_grad=True)  # 初始化 b 为零，并需要计算梯度
losses = np.zeros(500)  # 用来存储每次迭代的损失

lr = 0.005  # 学习率

# 训练模型的主循环
for ii in range(500):
    x, y = get_fake_data(batch_size=32)  # 每次迭代产生一个新的数据集

    # forward：计算预测值和损失
    y_pred = x.mm(w) + b.expand_as(y)  # 计算预测值 y_pred = w * x + b
    loss = 0.5 * (y_pred - y) ** 2  # 计算每个点的平方损失
    loss = loss.sum()  # 将所有点的损失加总
    losses[ii] = loss.item()  # 记录当前的损失

    # backward：自动计算梯度
    loss.backward()  # 计算相对于 w 和 b 的梯度

    # 更新参数
    w.data.sub_(lr * w.grad.data)  # 使用学习率更新权重 w
    b.data.sub_(lr * b.grad.data)  # 使用学习率更新偏置 b

    # 梯度清零
    w.grad.data.zero_()  # 每次更新参数后将梯度清零
    b.grad.data.zero_()  # 清零 b 的梯度

    if ii % 50 == 0:  # 每50次迭代绘制一次拟合曲线
        plt.clf()  # 清空当前图像
        x = t.arange(0, 6).float().view(-1, 1)  # 生成0到6之间的等差数列
        y = x.mm(w) + b.expand_as(x)  # 计算对应的y值（预测值）

        plt.plot(x.detach().numpy(), y.detach().numpy())  # 绘制拟合曲线

        x2, y2 = get_fake_data(batch_size=32)  # 生成新的数据
        plt.scatter(x2.detach().numpy(), y2.detach().numpy())  # 绘制真实数据的散点图

        plt.xlim(0, 5)  # 设置x轴的范围
        plt.ylim(0, 13)  # 设置y轴的范围
        plt.show(block=False)  # 非阻塞显示
        plt.pause(0.5)  # 暂停0.5秒

# 打印训练后的参数
print(f'w: {w.item():.3f}, b: {b.item():.3f}')

# 绘制损失随时间变化的曲线
plt.plot(losses)
plt.ylim(5, 50)  # 设置y轴的范围
plt.show()  # 显示损失曲线




