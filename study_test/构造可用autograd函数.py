# -*- coding: utf-8 -*-
# @Time : 2024/8/19 16:40
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : 构造可用autograd函数.py

#import
import torch as t

from torch.autograd import Function


class MultiplyAdd(Function):

    @staticmethod
    #前向传播
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x)  # 记录中间值
        output = w * x + b # 前向传播的函数
        return output

    @staticmethod
    #反向传播
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors  # 取出中间值
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b

x = t.ones(1)
w = t.rand(1, requires_grad = True)
b = t.rand(1, requires_grad = True)
# 开始前向传播
z = MultiplyAdd.apply(w, x, b)
# 开始反向传播
z.backward(retain_graph=True)
# x不需要求导，中间过程还是会计算它的导数，但随后被清空
print(x.grad, w.grad, b.grad)

# 调用MultiplyAdd.backward
# 输出grad_w, grad_x, grad_b
print(z.grad_fn)

result = z.grad_fn.apply(t.ones(1))
print(result)  # 输出与 z.grad_fn 相关的梯度传播









