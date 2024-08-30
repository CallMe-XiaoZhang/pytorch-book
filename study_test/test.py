# -*- coding: utf-8 -*-
# @Time : 2024/8/30 16:25
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : test.py
import torch

print(torch.cuda.device_count())
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.version.cuda)  # 检查 CUDA 版本