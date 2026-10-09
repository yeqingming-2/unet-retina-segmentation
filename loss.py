# -*- coding: utf-8 -*-
"""
loss.py —— 损失函数

损失函数衡量"模型预测"和"真实标注"差多远，训练就是不断让这个值变小。

这里提供两种：
  DiceLoss     —— 只算 Dice 损失
  DiceBCELoss  —— Dice 损失 + 二值交叉熵（本项目训练用的是这个，效果更稳）

Dice 系数是分割任务常用指标：两个集合重叠越多，Dice 越接近 1。
损失 = 1 - Dice，所以重叠越多损失越小。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """纯 Dice 损失"""
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 模型输出是 logit，先过 sigmoid 压到 0~1 变成概率
        inputs = torch.sigmoid(inputs)

        # 把多维张量拉平成 1 维，方便逐像素比较
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 交集 = 预测概率 × 真实标注 求和（像素都是血管才算交）
        intersection = (inputs * targets).sum()
        # Dice = 2*交集 / (预测和 + 标注和)，smooth=1 防止除零
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice   # 损失越小 = Dice 越高


class DiceBCELoss(nn.Module):
    """Dice 损失 + 二值交叉熵（BCE），混合损失，训练更稳定"""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 模型输出是 logit，先过 sigmoid 变成概率
        inputs = torch.sigmoid(inputs)

        # 拉平成一维逐像素比较
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算 Dice 损失（同上）
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # 二值交叉熵：衡量每个像素预测概率和真实值(0/1)的差异
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # 两者相加作为总损失
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
