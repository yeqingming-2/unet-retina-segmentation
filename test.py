# -*- coding: utf-8 -*-
"""
test.py —— 测试脚本（DRIVE / CHASE / HRF 通用）

用训练好的模型（files/checkpoint.pth）在测试集上推理：
  1. 对每张测试图做预测
  2. 计算 Jaccard / F1 / Recall / Precision / Accuracy 五个指标
  3. 把 原图|金标准|预测 三张横向拼成一张图，保存到 results/

运行方式（PyCharm 里直接右键 Run）：
    python test.py
"""
import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding

# 项目根目录（脚本所在目录），所有路径基于它，PyCharm 里直接运行不会报错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA = os.path.join(BASE_DIR, "new_data")
RESULT_DIR = os.path.join(BASE_DIR, "results")


def calculate_metrics(y_true, y_pred):
    """计算一张图的 5 个分割指标。
    y_true/y_pred 都是 (1, 512, 512) 的概率张量，>0.5 视为血管。"""
    # 真实标注：转 numpy，二值化(>0.5)，展平成一维
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # 预测结果：同样处理
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # 五个标准分割指标（sklearn 直接算）
    score_jaccard = jaccard_score(y_true, y_pred)      # IoU
    score_f1 = f1_score(y_true, y_pred)                # F1（Dice）
    score_recall = recall_score(y_true, y_pred)        # 召回率（血管找全没）
    score_precision = precision_score(y_true, y_pred)  # 精确率（找的准不准）
    score_acc = accuracy_score(y_true, y_pred)         # 像素准确率

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    """把单通道 (512,512) 的 mask 复制成 3 通道 (512,512,3)，
    这样能直接和彩色原图拼在一起显示。"""
    mask = np.expand_dims(mask, axis=-1)    # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


if __name__ == "__main__":
    """ 固定随机种子（与训练一致，保证复现） """
    seeding(42)

    """ 创建结果保存目录 """
    create_dir(RESULT_DIR)

    """ 读取测试集 """
    test_x = sorted(glob(os.path.join(NEW_DATA, "test", "image", "*")))
    test_y = sorted(glob(os.path.join(NEW_DATA, "test", "mask", "*")))

    """ 参数 """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = os.path.join(BASE_DIR, "files", "checkpoint.pth")

    """ 加载训练好的模型权重 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # 载入权重
    model.eval()   # 评估模式

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]   # 累加所有测试图的指标
    time_taken = []                              # 记录每张推理耗时（算 FPS 用）

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ 取出文件名（不带路径和扩展名） """
        name = os.path.splitext(os.path.basename(x))[0]

        """ 读取并预处理图像 """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)，BGR
        x = np.transpose(image, (2, 0, 1))       # 转成 (3, 512, 512)
        x = x / 255.0                            # 归一化
        x = np.expand_dims(x, axis=0)            # 加 batch 维 (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ 读取并预处理金标准 mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)   # (512, 512)
        y = np.expand_dims(mask, axis=0)             # (1, 512, 512)
        y = y / 255.0                                # 血管=1
        y = np.expand_dims(y, axis=0)                # (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ 预测 + 计时（算 FPS） """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)   # logit -> 概率 0~1
            total_time = time.time() - start_time
            time_taken.append(total_time)

            # 累加指标
            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            # 预测概率 -> 二值图（>0.5 是血管）
            pred_y = pred_y[0].cpu().numpy()     # (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)  # (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ 拼图：原图 | 金标准 | 预测，中间用灰线隔开 """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128   # 10 像素宽的灰色分隔线

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(os.path.join(RESULT_DIR, f"{name}.png"), cat_images)

    """ 输出平均指标（所有测试图取平均） """
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
