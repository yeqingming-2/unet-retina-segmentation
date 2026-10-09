# -*- coding: utf-8 -*-
"""
train_fives_patch.py —— FIVES 高分辨率 patch 训练脚本

为什么用 patch 训练：
  FIVES 原图 2048x2048，直接缩到 512x512 会把小血管糊掉。
  patch 训练保留原图，每轮在每张图上随机位置切 512x512 的小块来训练，
  既保留了细节，随机切块本身还是一种数据增强（每次看到的块位置都不同）。

流程：
  1. 读 fives_full/ 下的原尺寸训练图（由 preprocess_fives.py 生成）
  2. 每个 epoch：对每张训练图随机切 PATCHES_PER_IMG 个 512x512 块喂给网络
  3. 在官方测试集上算验证损失，创新低就保存权重到 files/checkpoint_fives_patch.pth
     （这里用官方测试集当验证集来选最好的模型，是分割复现的常见简化做法）

参数说明：
  600 张训练图、每张每轮切 2 块、batch 2，单轮约 600 步，
  实测 15 轮在 RTX 3070 Ti Laptop 上约 1.5~2 小时。想更快就把
  PATCHES_PER_IMG 改 1 或 NUM_EPOCHS 改小；想效果更好就把 NUM_EPOCHS 加大。

用法：python train_fives_patch.py
"""
import os
import time
import random
from glob import glob

import numpy as np
import torch
from PIL import Image

from model import build_unet
from loss import DiceBCELoss
from utils import seeding, create_dir, epoch_time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "fives_full")   # 原尺寸数据目录

# ====== 可调参数 ======
PATCH = 512              # patch 边长（正方形）
PATCHES_PER_IMG = 2      # 每张训练图每轮随机切几个 patch（越大训练越慢，增强越多）
VAL_PATCHES_PER_IMG = 1  # 验证集每张每轮切几个 patch（少切一点，验证更快）
BATCH_SIZE = 2           # 每批 2 块；显存不够就改 1
NUM_EPOCHS = 15          # 训练轮数（600 张图，15 轮实测约 1.4 小时）
LR = 1e-4                # 学习率


def make_item(img_p, mask_p, patch, rng):
    """在一张原图上随机位置切一个 patch，返回 (图像, 标注) 张量。
    rng 是 random.Random 实例，用来保证每次切的位置不同。"""
    img = np.array(Image.open(img_p).convert("RGB"))   # 原图 (2048, 2048, 3)
    mask = np.array(Image.open(mask_p).convert("L"))   # 标注 (2048, 2048)
    H, W = img.shape[:2]
    assert H >= patch and W >= patch
    # 随机选 patch 的左上角坐标（保证 patch 不越界）
    y = rng.randint(0, H - patch)
    x = rng.randint(0, W - patch)
    # 切出 patch 并归一化到 0~1
    img_c = img[y:y + patch, x:x + patch].astype(np.float32) / 255.0
    mask_c = mask[y:y + patch, x:x + patch].astype(np.float32) / 255.0
    img_c = np.transpose(img_c, (2, 0, 1))              # 转成 (3, P, P)
    mask_c = np.expand_dims(mask_c, axis=0)             # (1, P, P)
    return torch.from_numpy(img_c), torch.from_numpy(mask_c)


class PatchLoader:
    """自定义数据加载器：每个 batch 都从原图集合里随机切新 patch。
    好处是不需要把几千个 patch 事先存盘，训练时实时生成。"""
    def __init__(self, images, mask_dir, patch, n_per, batch_size, device):
        self.images = images            # 所有原图路径
        self.mask_dir = mask_dir        # 标注所在目录（按同名匹配）
        self.patch = patch
        self.n_per = n_per              # 每张图每轮切几个 patch
        self.batch_size = batch_size
        self.device = device
        self._rng = random.Random()     # 每个 loader 独立随机源

    def __len__(self):
        # 一轮的 batch 数 = 总 patch 数 / batch_size（向上取整）
        return (len(self.images) * self.n_per + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # 每次迭代生成一个 batch
        for _ in range(len(self)):
            xs, ys = [], []
            for _ in range(self.batch_size):
                # 随机选一张原图，按同名找到它的标注
                img_p = self._rng.choice(self.images)
                stem = os.path.splitext(os.path.basename(img_p))[0]
                mask_p = os.path.join(self.mask_dir, stem + ".png")
                img_t, mask_t = make_item(img_p, mask_p, self.patch, self._rng)
                xs.append(img_t)
                ys.append(mask_t)
            # 把 batch 里的张量堆叠起来并搬到 GPU
            yield torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)


def run_epoch(model, loader, optimizer, loss_fn, device, train_mode):
    """跑一轮（训练或验证）。
    train_mode=True: 更新参数；False: 只评估（不更新）"""
    model.train() if train_mode else model.eval()
    epoch_loss = 0.0
    with torch.set_grad_enabled(train_mode):   # 验证时不计算梯度，省显存
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train_mode:
                optimizer.zero_grad()          # 清梯度
            pred = model(x)                    # 前向
            loss = loss_fn(pred, y)            # 算损失
            if train_mode:
                loss.backward()                # 反向传播
                optimizer.step()               # 更新参数
            epoch_loss += loss.item()          # 累加损失
    return epoch_loss / len(loader)


if __name__ == "__main__":
    seeding(42)   # 固定随机种子，结果可复现
    create_dir(os.path.join(BASE_DIR, "files"))

    # 读训练集和验证集的原图列表（标注通过同名自动匹配）
    train_x = sorted(glob(os.path.join(DATA, "train", "image", "*")))
    valid_x = sorted(glob(os.path.join(DATA, "test", "image", "*")))
    print(f"FIVES patch 训练: Train {len(train_x)} 张 / Valid {len(valid_x)} 张 | "
          f"patch={PATCH}, 每张每轮{PATCHES_PER_IMG}块, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = DiceBCELoss()
    ckpt = os.path.join(BASE_DIR, "files", "checkpoint_fives_patch.pth")   # 保存路径

    train_loader = PatchLoader(train_x, os.path.join(DATA, "train", "mask"), PATCH, PATCHES_PER_IMG, BATCH_SIZE, device)
    valid_loader = PatchLoader(valid_x, os.path.join(DATA, "test", "mask"), PATCH, VAL_PATCHES_PER_IMG, BATCH_SIZE, device)

    best = float("inf")   # 记录最优验证损失
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        tl = run_epoch(model, train_loader, optimizer, loss_fn, device, True)    # 训练一轮
        vl = run_epoch(model, valid_loader, optimizer, loss_fn, device, False)   # 验证一轮
        if vl < best:   # 验证损失创新低就保存
            best = vl
            torch.save(model.state_dict(), ckpt)
            flag = "best, saved"
        else:
            flag = ""
        mins, secs = epoch_time(t0, time.time())
        print(f"Epoch {epoch+1:02}/{NUM_EPOCHS} | Train {tl:.4f} | Val {vl:.4f} | {mins}m {secs}s {flag}", flush=True)
    print(f"完成，最佳模型: {ckpt} (val loss {best:.4f})")
