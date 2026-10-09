# -*- coding: utf-8 -*-
"""
preprocess_drive.py —— 旧版 DRIVE 专用预处理脚本（仅 DRIVE，建议优先用 preprocess_all.py）

作用：把 DRIVE 原始图像(584x565)和金标准 mask 统一 resize 到 512x512，
     输出到 new_data/ 供 train.py / test.py 使用：
       new_data/train/image/*.png   (训练原图)
       new_data/train/mask/*.png    (训练金标准, 血管=255)
       new_data/test/image/*.png    (测试原图)
       new_data/test/mask/*.png     (测试金标准)

注意：这个脚本只能处理 DRIVE。要切换 CHASE / HRF，请用 preprocess_all.py。
"""
import os
from glob import glob
from PIL import Image

# 原始数据集路径（DRIVE），按你的实际路径修改
SRC = r"D:\eyedata\彩色眼底图数据库\DRIVE\DRIVE"
# 输出目录：项目根目录下的 new_data（自动定位，PyCharm 里直接运行即可）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(BASE_DIR, "new_data")
H = W = 512   # 统一尺寸


def process(img_dir, mask_dir, out_dir):
    """把 img_dir 下的图像和 mask_dir 下的标注一一对应，resize 后存到 out_dir"""
    os.makedirs(os.path.join(out_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
    images = sorted(glob(os.path.join(img_dir, "*")))
    masks = sorted(glob(os.path.join(mask_dir, "*")))
    assert len(images) == len(masks), f"图像与mask数量不一致: {len(images)} vs {len(masks)}"
    for img_p, mask_p in zip(images, masks):
        # 原图缩放到 512x512（双线性插值保细节）
        img = Image.open(img_p).convert("RGB")
        img = img.resize((W, H), Image.BILINEAR)
        name = os.path.splitext(os.path.basename(img_p))[0]
        img.save(os.path.join(out_dir, "image", name + ".png"))
        # 金标准缩放到 512x512（最近邻插值，保持二值不变形）
        mask = Image.open(mask_p).convert("L")
        mask = mask.resize((W, H), Image.NEAREST)
        mask.save(os.path.join(out_dir, "mask", name + ".png"))
    print(f"{out_dir}: {len(images)} 对 (image+mask) 完成")


if __name__ == "__main__":
    # 训练集：training/images ↔ training/1st_manual
    process(os.path.join(SRC, "training", "images"),
            os.path.join(SRC, "training", "1st_manual"),
            os.path.join(DST, "train"))
    # 测试集：test/images ↔ test/1st_manual
    process(os.path.join(SRC, "test", "images"),
            os.path.join(SRC, "test", "1st_manual"),
            os.path.join(DST, "test"))
    print("预处理完成")
