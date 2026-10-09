# -*- coding: utf-8 -*-
"""
data_aug.py —— 数据增强脚本（可选，本项目训练未使用）

作用：对 DRIVE 原始图像做 水平翻转/垂直翻转/旋转 三种增强，
     把每张图变成 4 张（原图 + 3 张增强），扩大训练数据量。

注意：
  1. 依赖 albumentations 库（当前环境未安装，运行前需 pip install albumentations）
  2. 底部数据路径是原作者 Linux 机器的路径，需改成你自己的
  3. 本项目训练用不到它：train.py 直接读取 new_data/ 下的数据即可。
     如果你想用增强后的数据训练，先跑本脚本生成增强数据，
     再把 train.py 的数据路径指向增强后的目录。
"""
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate


def create_dir(path):
    """创建目录（不存在才创建）"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    """读取 DRIVE 原始数据路径列表（图像和标注一一对应）"""
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)


def augment_data(images, masks, save_path, augment=True):
    """对每张图做增强并保存。
    augment=True: 生成 原图 + 水平翻转 + 垂直翻转 + 旋转45° 共 4 张
    augment=False: 只保留原图 1 张（测试集用）"""
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ 取文件名（去掉路径和扩展名） """
        name = os.path.splitext(os.path.basename(x))[0]

        """ 读图像和标注（标注是 gif，用 imageio 读） """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            # 水平翻转
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            # 垂直翻转
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            # 随机旋转（最多 45 度）
            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]   # 4 张增强后的图像
            Y = [y, y1, y2, y3]   # 对应的 4 张标注

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)   # 统一缩放到 512x512
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """ 固定随机种子（旋转的随机角度可复现） """
    np.random.seed(42)

    """ 读取数据（改成你自己的 DRIVE 路径） """
    data_path = "/media/nikhil/ML/ml_dataset/Retina blood vessel segmentation/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ 创建输出目录 """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """ 训练集做增强，测试集不增强 """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)
