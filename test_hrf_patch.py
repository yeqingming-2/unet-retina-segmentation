# -*- coding: utf-8 -*-
"""
test_hrf_patch.py —— HRF 滑窗推理测试脚本

作用：用 patch 训练好的模型（files/checkpoint_hrf_patch.pth）在 HRF 测试集上
     做完整推理。因为模型只能吃 512x512 的输入，而 HRF 原图是 3504x2336，
     所以用"滑窗"方式：把整张图切成一个个 512x512 的窗口依次预测，
     相邻窗口重叠 50%（stride=256），重叠区域取平均，最后拼回原尺寸。

输出：
  1. results_hrf/ 下每张测试图的 原图|金标准|预测 三连图（原尺寸）
  2. 打印 Jaccard / F1 / Recall / Precision / Acc 五个指标

用法：python test_hrf_patch.py
"""
import os
import time
from glob import glob

import numpy as np
import cv2
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import seeding, create_dir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "hrf_full")        # 原尺寸数据
RESULT_DIR = os.path.join(BASE_DIR, "results_hrf")  # 结果输出目录
CKPT = os.path.join(BASE_DIR, "files", "checkpoint_hrf_patch.pth")  # 模型权重

PATCH = 512      # 滑窗大小（和训练时一致）
STRIDE = 256     # 滑窗步长：256，即窗口之间重叠 50%（重叠处取平均，边缘更平滑）


def sliding_predict(model, img_np, device):
    """滑窗推理：把一张大图切成 512x512 窗口逐块预测，拼回原尺寸概率图。
    返回与原图同尺寸的 [0,1] 概率图。"""
    H, W = img_np.shape[:2]
    # 生成窗口起始位置：从 0 开始每 STRIDE 一个，并保证最后一个窗口贴到图边缘
    ys = list(range(0, H - PATCH + 1, STRIDE))
    xs = list(range(0, W - PATCH + 1, STRIDE))
    if ys[-1] != H - PATCH:
        ys.append(H - PATCH)
    if xs[-1] != W - PATCH:
        xs.append(W - PATCH)
    # acc: 每个像素累加的预测值；cnt: 每个像素被多少个窗口覆盖
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                # 取一个窗口，归一化并转成网络输入格式 (1, 3, 512, 512)
                crop = img_np[y0:y0 + PATCH, x0:x0 + PATCH].astype(np.float32) / 255.0
                crop = np.transpose(crop, (2, 0, 1))
                crop = torch.from_numpy(crop).unsqueeze(0).to(device)
                # 预测：sigmoid 变成概率，取第 0 张图的第 0 通道
                pred = torch.sigmoid(model(crop))[0, 0].cpu().numpy()
                acc[y0:y0 + PATCH, x0:x0 + PATCH] += pred   # 累加
                cnt[y0:y0 + PATCH, x0:x0 + PATCH] += 1.0    # 计数
    # 重叠区域取平均 → 最终概率图
    return acc / np.maximum(cnt, 1e-6)


def calculate_metrics(gt, pred):
    """计算单张图的 5 个指标（gt/pred 都是二维 0/1 图）"""
    gt_b = gt > 0.5
    pred_b = pred > 0.5
    return [jaccard_score(gt_b.ravel(), pred_b.ravel()),       # IoU
            f1_score(gt_b.ravel(), pred_b.ravel()),            # F1
            recall_score(gt_b.ravel(), pred_b.ravel()),        # 召回率
            precision_score(gt_b.ravel(), pred_b.ravel()),     # 精确率
            accuracy_score(gt_b.ravel(), pred_b.ravel())]      # 准确率


if __name__ == "__main__":
    seeding(42)
    create_dir(RESULT_DIR)

    # 读测试集（图像和标注同名配对）
    test_x = sorted(glob(os.path.join(DATA, "test", "image", "*")))
    test_y = sorted(glob(os.path.join(DATA, "test", "mask", "*")))
    assert len(test_x) == len(test_y), "图像与mask数量不一致"

    # 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    print(f"已加载 {CKPT}，测试 {len(test_x)} 张 (滑窗 {PATCH}x{PATCH}, stride {STRIDE})")

    metrics = [0.0] * 5   # 累加所有测试图的指标
    time_taken = []
    for i, (img_p, mask_p) in enumerate(zip(test_x, test_y)):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        # 读原图和金标准
        img = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(mask_p).convert("L")).astype(np.float32) / 255.0

        # 滑窗推理
        t0 = time.time()
        prob = sliding_predict(model, img, device)
        time_taken.append(time.time() - t0)

        # 概率 > 0.5 视为血管，算指标
        pred = (prob > 0.5).astype(np.uint8)
        score = calculate_metrics(mask, pred)
        metrics = [m + s for m, s in zip(metrics, score)]

        # 保存三连图：原图 | 金标准 | 预测（中间灰线分隔）
        mask3 = np.stack([mask, mask, mask], axis=-1)
        pred3 = np.stack([pred, pred, pred], axis=-1) * 255
        line = np.ones((img.shape[0], 10, 3), dtype=np.uint8) * 128
        cat = np.concatenate([img, line, mask3.astype(np.uint8) * 255, line, pred3], axis=1)
        cv2.imwrite(os.path.join(RESULT_DIR, f"{stem}.png"), cat)
        print(f"[{i+1}/{len(test_x)}] {stem} 完成")

    # 输出平均指标
    n = len(test_x)
    names = ["Jaccard", "F1", "Recall", "Precision", "Acc"]
    vals = [m / n for m in metrics]
    print("\n=== HRF 测试集指标 (滑窗拼接原尺寸) ===")
    for name, v in zip(names, vals):
        print(f"{name}: {v:.4f}")
    print(f"平均单张推理: {np.mean(time_taken):.2f}s")
