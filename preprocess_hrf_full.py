# -*- coding: utf-8 -*-
"""
preprocess_hrf_full.py —— HRF 高分辨率预处理脚本

作用：把 HRF 数据集按原始分辨率(3504x2336)做 train/test 划分，输出到 hrf_full/。
     与 preprocess_all.py 不同，这里【不缩放】图像，为 patch 训练保留细节。

为什么需要它：
  HRF 原图 3504x2336，直接缩到 512x512 会把小血管糊掉。
  patch 训练（train_hrf_patch.py）需要原尺寸的图，训练时随机切 512x512 小块。

用法：python preprocess_hrf_full.py
"""
import os
import random
import shutil
from glob import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = r"D:\eyedata\HRF"                        # 原始 HRF 数据目录
DST = os.path.join(BASE_DIR, "hrf_full")       # 输出目录
SEED = 42                                       # 固定种子，划分与 preprocess_all.py 一致
TRAIN_RATIO = 0.7                               # 70% 训练 / 30% 测试（与通用脚本一致）


def main():
    """读 HRF 图像，按 图像名 ↔ manual1/同名.tif 配对，划分并复制到 hrf_full/"""
    images = sorted(glob(os.path.join(SRC, "images", "*")))
    pairs = []
    for img_p in images:
        stem = os.path.splitext(os.path.basename(img_p))[0]   # 01_dr
        mask_p = os.path.join(SRC, "manual1", f"{stem}.tif")  # 金标准
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
    pairs.sort(key=lambda p: os.path.basename(p[0]))
    random.Random(SEED).shuffle(pairs)          # 固定种子打乱
    n = int(len(pairs) * TRAIN_RATIO)
    train_pairs, test_pairs = pairs[:n], pairs[n:]
    print(f"HRF: 训练 {len(train_pairs)} / 测试 {len(test_pairs)}")

    # 分别处理训练集和测试集
    for sub, sub_pairs in (("train", train_pairs), ("test", test_pairs)):
        img_dir = os.path.join(DST, sub, "image")
        mask_dir = os.path.join(DST, sub, "mask")
        # 先清空旧文件，避免残留
        for d in (img_dir, mask_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        # 原尺寸直接另存为 png（不做任何缩放）
        for img_p, mask_p in sub_pairs:
            stem = os.path.splitext(os.path.basename(img_p))[0]
            img = Image.open(img_p).convert("RGB")
            mask = Image.open(mask_p).convert("L")
            assert img.size == mask.size, f"尺寸不一致: {img_p} {img.size} vs {mask.size}"
            img.save(os.path.join(img_dir, stem + ".png"))
            mask.save(os.path.join(mask_dir, stem + ".png"))
        print(f"{sub}: {len(sub_pairs)} 张完成 (原尺寸 {img.size})")


if __name__ == "__main__":
    main()
