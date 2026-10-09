# -*- coding: utf-8 -*-
"""
preprocess_fives.py —— FIVES 高分辨率预处理脚本

作用：把 FIVES 数据集整理成 train/test 两套目录，输出到 fives_full/。
     FIVES 官方已经分好了训练集(600张)和测试集(200张)，
     图像和标注【同名同目录】（如 train/Original/1_A.png ↔ train/Ground truth/1_A.png），
     所以这里不需要配对和划分，直接原尺寸复制另存为 PNG 即可。

为什么不做缩放：
  FIVES 原图 2048x2048，直接缩到 512x512 会把细小血管糊掉。
  保持原尺寸，训练时用 patch（train_fives_patch.py）随机切小块，保留细节。

FIVES 数据说明：
  图像: Original/*.png     眼底彩照（RGB）
  标注: Ground truth/*.png 血管分割金标准（二值：255=血管，0=背景）
  文件名后缀: A=健康 / D=糖尿病 / G=青光眼 / N=正常（对分割任务只是类别标签，不影响训练）

用法：python preprocess_fives.py
"""
import os
import shutil
from glob import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = r"D:\eyedata\FIVES"                 # 原始 FIVES 数据目录
DST = os.path.join(BASE_DIR, "fives_full")  # 输出目录


def copy_split(sub):
    """把 FIVES 的一个子集（train 或 test）整理成 fives_full/<sub>/{image,mask}。
    图像和标注同名，直接按文件名一一对应。"""
    img_dir = os.path.join(SRC, sub, "Original")
    mask_dir = os.path.join(SRC, sub, "Ground truth")
    images = sorted(glob(os.path.join(img_dir, "*")))
    print(f"{sub}: 找到 {len(images)} 张原图")

    # 输出目录（先清空旧的，避免残留文件混进来）
    dst_img = os.path.join(DST, sub, "image")
    dst_mask = os.path.join(DST, sub, "mask")
    for d in (dst_img, dst_mask):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # 逐张复制：原尺寸另存为 PNG（不做任何缩放）
    for img_p in images:
        stem = os.path.splitext(os.path.basename(img_p))[0]          # 如 1_A
        mask_p = os.path.join(mask_dir, stem + ".png")               # 同名标注
        if not os.path.exists(mask_p):
            print(f"警告：{img_p} 没有对应标注 {mask_p}，跳过")
            continue
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")   # 金标准转灰度（0/255）
        assert img.size == mask.size, f"尺寸不一致: {img_p} {img.size} vs {mask.size}"
        img.save(os.path.join(dst_img, stem + ".png"))
        mask.save(os.path.join(dst_mask, stem + ".png"))
    print(f"{sub}: 完成 {len(images)} 张 (原尺寸 {img.size})")


if __name__ == "__main__":
    for sub in ("train", "test"):
        copy_split(sub)
    print(f"预处理完成，数据在 {DST}")
