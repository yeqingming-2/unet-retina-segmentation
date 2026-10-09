# -*- coding: utf-8 -*-
"""
data.py —— 数据集读取类

PyTorch 的 Dataset 类负责"给定一个序号 index，返回一份数据"：
  __getitem__: 按序号读取一张图像和它对应的标注 mask
  __len__:     返回数据集里有多少张图
配合 DataLoader 使用，DataLoader 会自动分批、打乱、多线程加载。
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    """DRIVE 风格的数据集：图像列表 + mask 列表，一一对应。"""
    def __init__(self, images_path, masks_path):
        # images_path: 所有图像文件路径的列表
        # masks_path:  所有标注文件路径的列表（与图像一一对应）
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)   # 样本总数

    def __getitem__(self, index):
        """读取第 index 张图及其标注，转成网络需要的张量格式"""

        """ 读取图像 """
        # cv2 按彩色读入：形状 (512, 512, 3)，数值范围 0~255
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0                 # 归一化到 0~1，利于网络训练
        image = np.transpose(image, (2, 0, 1))  # 转成 PyTorch 习惯的 (通道, 高, 宽) = (3, 512, 512)
        image = image.astype(np.float32)      # 网络用 float32
        image = torch.from_numpy(image)

        """ 读取标注 mask """
        # 灰度读入：形状 (512, 512)，像素 0(背景) 或 255(血管)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0                   # 归一化到 0~1，血管=1，背景=0
        mask = np.expand_dims(mask, axis=0)   # 加一个通道维 → (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask   # 返回一对 (图像, 标注)

    def __len__(self):
        return self.n_samples
