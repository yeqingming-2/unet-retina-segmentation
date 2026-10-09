# -*- coding: utf-8 -*-
"""
model.py —— U-Net 网络结构定义

U-Net 的结构可以理解为"编码器-解码器"：
  编码器（左半边）：逐层下采样（池化），把图像压缩成越来越抽象的特征图
  解码器（右半边）：逐层上采样（转置卷积），把特征图恢复回原图大小
  跳跃连接（skip）：解码器每一层都拼上编码器对应层的特征，帮助恢复细节

输入：彩色眼底图 (batch, 3, 512, 512)
输出：预测图 (batch, 1, 512, 512)，每个像素是"该像素属于血管"的得分（未经过 sigmoid）
"""
import torch
import torch.nn as nn


class conv_block(nn.Module):
    """卷积块：连续两次 [卷积 + 批归一化 + ReLU]。
    这是 U-Net 里最基础的特征提取单元。"""
    def __init__(self, in_c, out_c):
        """
        in_c: 输入通道数
        out_c: 输出通道数
        """
        super().__init__()

        # 第一次卷积：把输入通道数 in_c 变成 out_c
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)   # 批归一化：加速训练、稳定梯度

        # 第二次卷积：通道数不变，继续提取特征
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()   # 激活函数：给网络引入非线性

    def forward(self, inputs):
        # 第一次 卷积 -> 归一化 -> 激活
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        # 第二次 卷积 -> 归一化 -> 激活
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    """编码器块：一个卷积块 + 一个最大池化。
    卷积块负责提取特征，池化把尺寸缩小一半（下采样）。
    返回值有两个：
      x —— 卷积后的特征图（留给解码器的跳跃连接用）
      p —— 池化后的特征图（送入下一层编码器）"""
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)          # 特征提取
        self.pool = nn.MaxPool2d((2, 2))             # 2x2 最大池化，宽高各减半

    def forward(self, inputs):
        x = self.conv(inputs)    # 先卷积提特征
        p = self.pool(x)         # 再池化缩小尺寸

        return x, p


class decoder_block(nn.Module):
    """解码器块：上采样 + 拼接跳跃连接 + 卷积块。
    上采样把特征图尺寸放大一倍，然后和编码器对应层的特征拼在一起，
    最后用卷积块融合。"""
    def __init__(self, in_c, out_c):
        super().__init__()

        # 转置卷积（也叫反卷积）：尺寸放大一倍，通道 in_c -> out_c
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        # 拼接后通道数 = out_c（上采样结果） + out_c（跳跃连接来的） = 2*out_c
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)                              # 上采样放大
        x = torch.cat([x, skip], axis=1)                 # 沿通道维拼接跳跃连接的特征
        x = self.conv(x)                                 # 卷积融合
        return x


class build_unet(nn.Module):
    """完整的 U-Net 网络。

    编码器（下采样路径）：
      e1: 3 -> 64    e2: 64 -> 128   e3: 128 -> 256   e4: 256 -> 512
    瓶颈（最底层，特征最抽象）：
      b: 512 -> 1024
    解码器（上采样路径）：
      d1: 1024 -> 512   d2: 512 -> 256   d3: 256 -> 128   d4: 128 -> 64
    输出层：64 -> 1（每个像素一个值，表示血管概率的 logit）"""
    def __init__(self):
        super().__init__()

        """ 编码器：通道数逐层翻倍，特征图尺寸逐层减半 """
        self.e1 = encoder_block(3, 64)      # 输入是 RGB 三通道
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ 瓶颈：最底层 """
        self.b = conv_block(512, 1024)

        """ 解码器：通道数逐层减半，特征图尺寸逐层恢复 """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ 输出层：1x1 卷积把 64 通道压成 1 通道（血管/背景二分类） """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ 编码器：逐层下采样，同时保存每层特征用于跳跃连接
            s1~s4 是各层卷积后的特征图，p1~p4 是池化后的结果 """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ 瓶颈 """
        b = self.b(p4)

        """ 解码器：逐层上采样，并拼接对应层的跳跃连接特征 """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)   # 输出与输入同尺寸 (batch, 1, H, W)

        return outputs


if __name__ == "__main__":
    # 简单自测：造一个随机输入，看输出形状是否正确
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)   # 期望 (2, 1, 512, 512)
