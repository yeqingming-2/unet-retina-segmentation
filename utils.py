# -*- coding: utf-8 -*-
"""
utils.py —— 工具函数

三个小工具：
  seeding()    固定所有随机种子，保证每次训练结果可复现
  create_dir() 创建目录（不存在才创建）
  epoch_time() 计算一个 epoch 花了多长时间
"""
import os
import time
import random
import numpy as np
import torch


def seeding(seed):
    """固定随机种子：让随机数生成方式完全确定，
    这样每次重跑代码，数据划分、初始化、训练结果都一致。"""
    random.seed(seed)                          # Python 内置随机
    os.environ["PYTHONHASHSEED"] = str(seed)   # 哈希随机种子
    np.random.seed(seed)                       # numpy 随机
    torch.manual_seed(seed)                    # CPU 上的 torch 随机
    torch.cuda.manual_seed(seed)               # GPU 上的 torch 随机
    torch.backends.cudnn.deterministic = True  # 让 cuDNN 也确定化（略慢但可复现）


def create_dir(path):
    """创建目录；如果目录已经存在就什么都不做"""
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time, end_time):
    """把耗时秒数转成 (分钟, 秒)，方便打印"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
