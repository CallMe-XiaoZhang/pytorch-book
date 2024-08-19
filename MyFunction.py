# -*- coding: utf-8 -*-
# @Time : 2024/8/19 17:47
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : MyFunction.py

#import
import numpy as np
import torch

def same_seed(seed):
    """
    设置随机种子以确保实验的可重复性。
    """
    torch.backends.cudnn.deterministic = True  # 确保每次运行结果一致
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
    print(f'Set Seed = {seed}')  # 输出确认信息

