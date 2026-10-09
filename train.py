# -*- coding: utf-8 -*-
"""
train.py —— 训练脚本（DRIVE / CHASE / HRF / FIVES 通用）

流程：
  1. 读取 new_data/ 下预处理好的训练集和测试集
  2. 构建 U-Net 模型、优化器、损失函数
  3. 训练 num_epochs 轮，每轮在训练集上学一遍、在验证集上评估一遍
  4. 验证集损失变小就保存模型权重到 files/checkpoint.pth

运行方式（PyCharm 里直接右键 Run，或在终端执行）：
    python train.py                     # 默认参数，等价于 DRIVE 的 50 轮 / batch 2
    python train.py --epochs 30         # 数据量大的数据集(如 FIVES 600 张)轮数可调小
    python train.py --epochs 30 --batch 4   # 显存够就加大 batch，训练更快

说明：
  - DRIVE(20 张训练图)：默认 50 轮约 9 分钟
  - FIVES(600 张训练图)：建议 --epochs 30 --batch 4，约 1 小时
"""
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader

from data import DriveDataset
from model import build_unet
from loss import DiceBCELoss
from utils import seeding, create_dir, epoch_time

# 项目根目录（脚本所在目录），所有路径基于它，PyCharm 里直接运行不会报错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA = os.path.join(BASE_DIR, "new_data")


def train(model, loader, optimizer, loss_fn, device):
    """训练一个 epoch：把整个训练集过一遍，更新模型参数"""
    epoch_loss = 0.0

    model.train()   # 切换到训练模式（启用 Dropout、BatchNorm 统计更新）
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)   # 图像移到 GPU
        y = y.to(device, dtype=torch.float32)   # 标注移到 GPU

        optimizer.zero_grad()       # 清空上一步的梯度
        y_pred = model(x)           # 前向：模型预测
        loss = loss_fn(y_pred, y)   # 计算损失（预测与真值的差距）
        loss.backward()             # 反向传播：算每个参数的梯度
        optimizer.step()            # 按梯度更新参数（学习一步）

        epoch_loss += loss.item()   # 累加损失，最后取平均

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    """验证一个 epoch：只评估不更新参数（no_grad 省显存、加速）"""
    epoch_loss = 0.0

    model.eval()    # 切换到评估模式
    with torch.no_grad():   # 不计算梯度，节省显存
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    """ 固定随机种子：保证每次运行结果一致，方便复现 """
    seeding(42)

    """ 创建保存模型的目录 """
    create_dir(os.path.join(BASE_DIR, "files"))

    """ 命令行参数（可选的）：不传就用默认值，行为和原来完全一样 """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（数据量大时调小，如 30）")
    parser.add_argument("--batch", type=int, default=2, help="batch size（显存够就调大，如 4，训练更快）")
    args = parser.parse_args()

    """ 读取数据：new_data 下 train(训练) / test(验证) 两个目录 """
    train_x = sorted(glob(os.path.join(NEW_DATA, "train", "image", "*")))
    train_y = sorted(glob(os.path.join(NEW_DATA, "train", "mask", "*")))

    valid_x = sorted(glob(os.path.join(NEW_DATA, "test", "image", "*")))
    valid_y = sorted(glob(os.path.join(NEW_DATA, "test", "mask", "*")))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ 超参数（想调整就改这里） """
    H = 512          # 图像高（预处理已统一成 512）
    W = 512          # 图像宽
    size = (H, W)
    batch_size = args.batch   # 每批几张；显存不够(OOM)就改成 1
    num_epochs = args.epochs  # 训练轮数
    lr = 1e-4        # 学习率
    checkpoint_path = os.path.join(BASE_DIR, "files", "checkpoint.pth")

    """ 构建数据集和加载器 """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 训练集打乱顺序，避免模型记住顺序
        num_workers=2       # 2 个进程并行读图，加快速度
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,      # 验证集不打乱
        num_workers=2
    )

    device = torch.device('cuda')   # 用 GPU 训练（你的电脑有 RTX 3070 Ti）
    model = build_unet()            # 创建 U-Net
    model = model.to(device)        # 模型搬到 GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # Adam 优化器
    # 学习率调度：验证集损失连续 5 轮不降，学习率自动减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()   # Dice + BCE 混合损失

    """ 开始训练 """
    best_valid_loss = float("inf")   # 记录最优验证损失，初始无穷大

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ 验证损失创新低就保存模型 """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)   # 只保存权重

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
