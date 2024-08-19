# -*- coding: utf-8 -*-
# @Time : 2024/8/19 15:02
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 自动求导.py

print('\n'*5)

#import
import torch as t

# #code
# #构造初始张量
# a = t.Tensor([1, 2, 3])
# a.requires_grad=True
# b = t.Tensor([4, 5, 6]).requires_grad_()
# #函数
# c = a**2 + b
# print(c)
# #反向传播计算梯度
# t.autograd.backward(c,grad_tensors=t.ones_like(c))
# #查看梯度
# print(a.grad)
# print(b.grad)
#
# print('\n'*2)
#
# #叶子张量
# print(a.is_leaf,b.is_leaf,c.is_leaf)

#
# x = t.ones(1)
# b = t.rand(1, requires_grad=True)#创建
# w = t.rand(1, requires_grad=True)#创建
# y = w * x  # 等价于y=w.mul(x)
# z = y + b  # 等价于z=y.add(b)
# #等价于 z = w*x+b
#
# print( x.requires_grad, b.requires_grad, w.requires_grad)
#
# # grad_fn可以查看这个Tensor的反向传播函数，
# # z是add函数的输出，所以它的反向传播函数是AddBackward
# print(z.grad_fn )
#
# # next_functions保存grad_fn的输入，grad_fn的输入是一个tuple
# # 第一个是y，它是乘法（mul）的输出，所以对应的反向传播函数y.grad_fn是MulBackward
# # 第二个是b，它是叶子节点，需要求导，所以对应的反向传播函数是AccumulateGrad
# print(z.grad_fn.next_functions)
#
# # 第一个是w，叶子节点，需要求导，梯度是累加的
# # 第二个是x，叶子节点，不需要求导，所以为None
# print(y.grad_fn.next_functions)
#
# # 叶子节点的grad_fn是None
# print(w.grad_fn, x.grad_fn)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# print('\n'*15)




# 使用retain_graph来保存buffer,不指定retain_graph=True，第二次计算梯度会报错
# z.backward(retain_graph=True)
# print(w.grad)
# # 多次反向传播，梯度累加，这也就是w中AccumulateGrad标识的含义
# z.backward()
# print(w.grad)









# #PyTorch使用的是动态图，因为它的计算图在每次前向传播时都是从头开始构建的，
# # 所以它能够使用Python控制语句（如for、if等）根据需求创建计算图。
#
# def abs(x):
#     if x.data[0] > 0:
#         return x
#     else:
#         return -x
#
# x = t.ones(1, requires_grad=True)
# y = abs(x)
# print(y.backward())
# print(x.grad)


t.set_grad_enabled(False)  # 更改了默认设置
x = t.ones(1)
w = t.rand(1, requires_grad=True)
y = x * w
# y依赖于w和x，虽然w.requires_grad=True，但是y的requires_grad依旧为False
print(x.requires_grad, w.requires_grad, y.requires_grad)
# 恢复默认配置
t.set_grad_enabled(True)
m = t.ones(1)
l = t.rand(1, requires_grad=True)
z = m * l
print(m.requires_grad, l.requires_grad, z.requires_grad)
























