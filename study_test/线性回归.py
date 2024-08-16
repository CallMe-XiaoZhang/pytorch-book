import torch as t
from matplotlib import pyplot as plt
from IPython import display

# 设置设备为CPU
device = 'cpu'

# 如果有GPU，则优先使用GPU
# device = 'cuda' if t.cuda.is_available() else 'cpu'

# 设置随机数种子，保证在不同电脑上运行时输出一致
t.manual_seed(1000)


def get_fake_data(batch_size=8):
    '''
    产生随机数据：y = 2*x + 3，并加上一些噪声
    Args:
        batch_size: 数据批量大小
    Returns:
        随机生成的x和y数据
    '''
    x = t.rand(batch_size, 1, device=device) * 5  # 生成0到5之间的随机数
    y = x * 2 + 3 + t.randn(batch_size, 1, device=device)  # y = 2x + 3 + 噪声
    return x, y


# 生成一批数据并绘制其分布
x, y = get_fake_data(batch_size=16)
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())  # 绘制散点图

# 随机初始化参数w和b
w = t.rand(1, 1).to(device)  # 权重初始化
b = t.zeros(1, 1).to(device)  # 偏置初始化

lr = 0.02  # 学习率learning rate

# 进行500次迭代
for ii in range(500):
    x, y = get_fake_data(batch_size=4)  # 每次迭代生成一批新的数据

    # 前向传播：计算预测值和损失
    y_pred = x.mm(w) + b.expand_as(y)  # 计算预测值 y_pred = x * w + b
    loss = 0.5 * (y_pred - y) ** 2  # 计算均方误差 (MSE)
    loss = loss.mean()  # 求均方误差的均值

    # 反向传播：手动计算梯度
    dloss = 1  # 损失函数对自身的导数
    dy_pred = dloss * (y_pred - y)  # 计算损失函数对预测值的导数

    dw = x.t().mm(dy_pred)  # 计算损失函数对w的导数
    db = dy_pred.sum()  # 计算损失函数对b的导数

    # 更新参数
    w.sub_(lr * dw)  # 使用梯度下降法更新权重w
    b.sub_(lr * db)  # 使用梯度下降法更新偏置b

    if ii % 50 == 0:  # 每50次迭代绘制一次拟合曲线
        display.clear_output(wait=True)  # 清空当前输出
        x = t.arange(0, 6).float().view(-1, 1)  # 生成0到6之间的等差数列
        y = x.mm(w) + b.expand_as(x)  # 计算对应的y值（预测值）

        plt.plot(x.cpu().numpy(), y.cpu().numpy())  # 绘制拟合曲线

        x2, y2 = get_fake_data(batch_size=32)  # 生成新的数据
        plt.scatter(x2.numpy(), y2.numpy())  # 绘制真实数据的散点图

        plt.xlim(0, 5)  # 设置x轴的范围
        plt.ylim(0, 13)  # 设置y轴的范围
        plt.show()  # 显示绘制的图像
        plt.pause(0.5)  # 暂停0.5秒

# 输出训练后的参数w和b
print('w: ', w.item(), 'b: ', b.item())
# test
#test
