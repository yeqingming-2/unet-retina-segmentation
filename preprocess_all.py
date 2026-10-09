# -*- coding: utf-8 -*-
"""
preprocess_all.py —— 通用数据预处理脚本（DRIVE / CHASE / HRF / FIVES 四个数据集切换）

作用：把原始眼底数据集统一处理成 512x512 的 PNG，输出到 new_data/，
      train.py / test.py 直接读 new_data/ 就能训练测试。

用法（PyCharm 里右键 Run with Parameters，或在终端执行）：
    python preprocess_all.py --dataset drive   # 切到 DRIVE
    python preprocess_all.py --dataset chase   # 切到 CHASE_DB1
    python preprocess_all.py --dataset hrf     # 切到 HRF
    python preprocess_all.py --dataset fives   # 切到 FIVES（自带 train/test 划分，图像与标注同名）

为什么不能只改路径（四个数据集的差异，脚本已自动处理）：
  1) 文件匹配规则：CHASE 图像名是 Image_01L.jpg，标注名是 Image_01L_1stHO.png，
     两者名字不同，必须按前缀配对；DRIVE 按数字前缀配对（21_training.tif ↔ 21_manual1.gif）
  2) 训练/测试划分：DRIVE/FIVES 自带 train/test 目录；CHASE/HRF 没有划分，
     脚本按 TRAIN_RATIO=70% 随机切分（固定种子，可复现）
  3) 标注选择：CHASE 每张图有两个标注(1stHO/2ndHO)，这里默认用 1stHO
  4) FIVES：图像与标注同名同目录结构（train/Original ↔ train/Ground truth），
     原图 2048x2048，这里统一缩放成 512x512（想保留原分辨率可改用 patch 训练方案）
"""
import os
import random
import shutil
from glob import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(BASE_DIR, "new_data")   # 输出目录
H = W = 512                                 # 统一尺寸
SEED = 42                                   # 固定随机种子，保证每次划分一致、结果可复现
TRAIN_RATIO = 0.7                           # 无官方划分的数据集(CHASE/HRF)：70% 训练 / 30% 测试


def process_pairs(pairs, out_dir):
    """把一组 (图像路径, 标注路径) 全部 resize 成 512x512 存到 out_dir。
    out_dir 下会生成 image/ 和 mask/ 两个子目录。"""
    os.makedirs(os.path.join(out_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
    for img_p, mask_p in pairs:
        # 读原图并缩放到 512x512（双线性插值，保细节）
        img = Image.open(img_p).convert("RGB")
        img = img.resize((W, H), Image.BILINEAR)
        # 读标注并缩放到 512x512（最近邻插值，保持 0/1 二值不变形）
        mask = Image.open(mask_p).convert("L")
        mask = mask.resize((W, H), Image.NEAREST)
        # 用图像原名作为输出文件名
        name = os.path.splitext(os.path.basename(img_p))[0]
        img.save(os.path.join(out_dir, "image", name + ".png"))
        mask.save(os.path.join(out_dir, "mask", name + ".png"))
    print(f"{out_dir}: {len(pairs)} 对完成")


def build_pairs(dataset):
    """按数据集构造 (图像路径, 标注路径) 列表。
    返回 (训练对列表, 测试对列表)。"""
    if dataset == "drive":
        # DRIVE：自带 training/ 和 test/ 两个目录，按文件名数字前缀配对
        # 例：training/images/21_training.tif ↔ training/1st_manual/21_manual1.gif
        src = r"D:\eyedata\彩色眼底图数据库\DRIVE\DRIVE"

        def build_split(img_sub, gt_sub):
            img_dir = os.path.join(src, img_sub, "images")
            gt_dir = os.path.join(src, gt_sub, "1st_manual")
            pairs = []
            for f in sorted(os.listdir(img_dir)):
                num = f.split("_")[0]                       # 文件名前缀数字，如 21
                gt_f = f"{num}_manual1.gif"                 # 对应标注名
                gt_p = os.path.join(gt_dir, gt_f)
                if os.path.exists(gt_p):
                    pairs.append((os.path.join(img_dir, f), gt_p))
            return pairs

        return build_split("training", "training"), build_split("test", "test")

    if dataset == "chase":
        # CHASE_DB1：28 张图，没有划分；图像 Image_01L.jpg ↔ 标注 Image_01L_1stHO.png
        src = r"D:\eyedata\CHASE_DB1"
        images = sorted(glob(os.path.join(src, "Image_*.jpg")))
        pairs = []
        for img_p in images:
            stem = os.path.splitext(os.path.basename(img_p))[0]     # Image_01L
            mask_p = os.path.join(src, f"{stem}_1stHO.png")         # 对应的 1stHO 标注
            if os.path.exists(mask_p):
                pairs.append((img_p, mask_p))
        pairs.sort(key=lambda p: os.path.basename(p[0]))            # 按文件名排序
        random.Random(SEED).shuffle(pairs)                          # 固定种子打乱
        n = int(len(pairs) * TRAIN_RATIO)                           # 前 70% 训练
        return pairs[:n], pairs[n:]

    if dataset == "hrf":
        # HRF：45 张图，没有划分；图像 01_dr.JPG ↔ 标注 manual1/01_dr.tif
        src = r"D:\eyedata\HRF"
        images = sorted(glob(os.path.join(src, "images", "*")))
        pairs = []
        for img_p in images:
            stem = os.path.splitext(os.path.basename(img_p))[0]     # 01_dr
            mask_p = os.path.join(src, "manual1", f"{stem}.tif")    # 金标准
            if os.path.exists(mask_p):
                pairs.append((img_p, mask_p))
        pairs.sort(key=lambda p: os.path.basename(p[0]))
        random.Random(SEED).shuffle(pairs)
        n = int(len(pairs) * TRAIN_RATIO)
        return pairs[:n], pairs[n:]

    if dataset == "fives":
        # FIVES：自带 train/test 划分；图像与标注同名同结构
        # 例：train/Original/1_A.png ↔ train/Ground truth/1_A.png
        src = r"D:\eyedata\FIVES"

        def build_split(split):
            img_dir = os.path.join(src, split, "Original")
            gt_dir = os.path.join(src, split, "Ground truth")
            pairs = []
            for f in sorted(os.listdir(img_dir)):
                gt_p = os.path.join(gt_dir, f)          # 标注与图像同名
                if os.path.exists(gt_p):
                    pairs.append((os.path.join(img_dir, f), gt_p))
            return pairs

        return build_split("train"), build_split("test")

    raise ValueError(f"未知数据集: {dataset}，仅支持 drive/chase/hrf/fives")


if __name__ == "__main__":
    import argparse
    # 命令行参数：--dataset 指定用哪个数据集
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["drive", "chase", "hrf", "fives"])
    args = parser.parse_args()

    train_pairs, test_pairs = build_pairs(args.dataset)
    print(f"数据集 {args.dataset}: 训练 {len(train_pairs)} 对 / 测试 {len(test_pairs)} 对")

    # 关键步骤：先清空 old new_data，避免多个数据集的文件混在一起
    # （不删的话，DRIVE 和 CHASE 的文件会同时存在，训练时数据就乱了）
    for sub in ("train", "test"):
        sub_dir = os.path.join(DST, sub)
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir)

    process_pairs(train_pairs, os.path.join(DST, "train"))
    process_pairs(test_pairs, os.path.join(DST, "test"))
    print("预处理完成")
