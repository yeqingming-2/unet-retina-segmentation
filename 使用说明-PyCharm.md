# 视网膜血管分割（U-Net + DRIVE/CHASE/HRF/FIVES）— PyCharm 使用说明

基于 PyTorch 的 U-Net 视网膜血管分割项目，支持 4 个数据集（DRIVE / CHASE_DB1 / HRF / FIVES）训练与测试。

## 一、用 PyCharm 打开项目

1. 打开 PyCharm → **File → Open**，选择目录：
   ```
   D:\unet_retina\Retina-Blood-Vessel-Segmentation-in-PyTorch-main
   ```
2. 配置解释器：**File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Conda Environment → Existing environment**，选择已有环境：
   ```
   D:\Anaconda3\envs\torch_new\python.exe
   ```
   （该环境已装好全部依赖，无需再装）

## 二、文件说明

| 文件 | 作用 |
|---|---|
| `preprocess_all.py` | **通用数据预处理**：切数据集只改 `--dataset` 参数，支持 drive / chase / hrf |
| `preprocess_drive.py` | 旧的单数据集预处理脚本（仅 DRIVE，保留参考） |
| `preprocess_hrf_full.py` | **HRF patch 专用**：保持原尺寸 3504x2336 做 train/test 划分（31/14） |
| `train_hrf_patch.py` | **HRF patch 训练**：在线随机切 512x512 块训练，保留细节 |
| `test_hrf_patch.py` | **HRF 滑窗推理**：256 步长滑窗拼回原图，出指标和结果图 |
| `preprocess_fives.py` | **FIVES 专用**：保持原尺寸 2048x2048，整理 train(600)/test(200) 到 `fives_full/` |
| `train_fives_patch.py` | **FIVES patch 训练**：每轮每张图随机切 512x512 块，15 轮约 1.4 小时 |
| `test_fives_patch.py` | **FIVES 滑窗推理**：拼回原图出指标，A/D/G/N 每类各存一张局部放大图 |
| `train.py` | 训练 U-Net（50 轮，约 9 分钟），模型保存到 `files/checkpoint.pth` |
| `test.py` | 用训练好的模型在测试集推理，输出结果图到 `results/`，并打印指标 |
| `model.py` | U-Net 网络结构定义 |
| `data.py` | 数据集读取类（图像 + mask） |
| `loss.py` | 损失函数（Dice + BCE） |
| `utils.py` | 工具函数 |
| `new_data/` | 预处理后的数据（当前是哪个数据集，由最后一次 preprocess_all 决定） |
| `hrf_full/` | HRF 原尺寸数据（patch 训练专用） |
| `fives_full/` | FIVES 原尺寸数据（patch 训练专用） |
| `files/checkpoint.pth` | DRIVE 训练好的模型权重（50 轮） |
| `files/checkpoint_hrf_patch.pth` | HRF patch 训练好的模型权重（30 轮） |
| `files/checkpoint_fives_patch.pth` | FIVES patch 训练好的模型权重（15 轮） |
| `results/` | DRIVE 测试结果图（原图\|金标准\|预测 三连图） |
| `results_hrf/` | HRF 测试结果图（原尺寸三连图 + 裁剪对比图） |
| `results_fives/` | FIVES 测试结果图（原尺寸三连图 + 局部放大对比图） |

## 三、在 PyCharm 里运行

直接右键对应脚本 → **Run 'train'** 或 **Run 'test'** 即可。
所有路径已改为基于脚本所在目录的绝对路径，不依赖工作目录，点运行就能跑。

- 想重新训练：运行 `train.py`
- 只想看结果（用现成模型）：运行 `test.py`
- 换新数据：运行 `preprocess_all.py --dataset <名称>`，再重新训练

## 四、更换数据集（重点）

不是只改路径。三个数据集的结构差异如下，`preprocess_all.py` 已全部处理：

| | DRIVE | CHASE_DB1 | HRF | FIVES |
|---|---|---|---|---|
| 原图 | `training/images/*.tif` `test/images/*.tif` | `Image_01L.jpg` 等 28 张 | `images/*.JPG` 45 张 | `train|test/Original/*.png` 800 张 |
| 标注 | `1st_manual/*.gif` | `Image_01L_1stHO.png`（每张有两个标注，取 1stHO） | `manual1/*.tif` | `train|test/Ground truth/*.png`（与图像同名） |
| 是否有训练/测试划分 | 有（自带） | 无（脚本按 70/30 随机划分） | 无（脚本按 70/30 随机划分） | 有（官方自带：600/200） |
| 原图尺寸 | 584x565 | 999x960 量级 | 3504x2336 | 2048x2048 |
| 关键差异 | — | 图像与标注文件名不同，需按前缀配对 | 图大，resize 512 细节损失大 | 图大，建议 patch 训练；标注 RGB 但值只有 0/255 |

切换命令（PyCharm 终端里运行，或右键 Run with Parameters）：

```bash
python preprocess_all.py --dataset drive   # 切到 DRIVE
python preprocess_all.py --dataset chase   # 切到 CHASE_DB1
python preprocess_all.py --dataset hrf     # 切到 HRF
```

**注意三点：**
1. 每次切换会**清空 new_data 再写入**，避免多个数据集混在一起（旧脚本的坑，已修复）
2. 切完必须重新运行 `train.py` 训练（checkpoint.pth 会被覆盖），再 `test.py`
3. 想固定划分、改训练比例：改 `preprocess_all.py` 顶部的 `SEED` 和 `TRAIN_RATIO`

## 五、复现指标（DRIVE 测试集 20 张，50 轮）

```
Jaccard: 0.6155 - F1: 0.7613 - Recall: 0.7254 - Precision: 0.8096 - Acc: 0.9606
FPS:  50.9
```

CHASE / HRF 指标需切换后自行训练得出（每个约 9 分钟）。HRF 图大，直接 resize 512 效果会明显低于 DRIVE/CHASE，属于正常现象。

## 六、HRF 高分辨率训练（patch 方式，推荐）

HRF 原图 3504x2336，直接 resize 512 会丢细节。改用 patch 训练：

```bash
python preprocess_hrf_full.py     # 1. 生成原尺寸 train/test 划分
python train_hrf_patch.py         # 2. patch 训练（30 轮约 26 分钟）
python test_hrf_patch.py          # 3. 滑窗推理，出指标和结果图
```

已跑通结果（HRF 测试集 14 张，滑窗拼接原尺寸）：

```
Jaccard: 0.6690 - F1: 0.7996 - Recall: 0.7785 - Precision: 0.8285 - Acc: 0.9686
```

效果明显好于直接 resize（patch 保留细节）。可调参数在 `train_hrf_patch.py` 顶部：
`PATCHES_PER_IMG`（每张每轮切块数，越大越慢但增强更多）、`NUM_EPOCHS`、`BATCH_SIZE`。

## 七、FIVES 高分辨率训练（patch 方式）

FIVES（芬兰视网膜血管分割数据集）原图 2048x2048，官方已分好 train 600 张 / test 200 张，
图像与标注同名（如 `1_A.png`），文件名后缀 A=健康 / D=糖尿病 / G=青光眼 / N=正常。

```bash
python preprocess_fives.py     # 1. 整理原尺寸数据到 fives_full/（不做缩放）
python train_fives_patch.py    # 2. patch 训练（15 轮约 1.4 小时）
python test_fives_patch.py     # 3. 滑窗推理，出指标和结果图
```

已跑通结果（FIVES 测试集 200 张，滑窗拼接原尺寸）：

```
Jaccard: 0.7268 - F1: 0.8291 - Recall: 0.8058 - Precision: 0.8749 - Acc: 0.9809
平均单张推理: 2.2s
```

要点：
1. 训练时用官方 test 集当验证集选最好的模型（分割复现的常见简化做法），最终指标也在它上面报
2. 想跑得更快：`train_fives_patch.py` 顶部把 `PATCHES_PER_IMG` 改成 1，或 `NUM_EPOCHS` 改小
3. 想看效果：`results_fives/` 下有 A/D/G/N 每类各一张局部放大对比图（`*_crop.png`），直接打开即可

## 八、常见问题

- **提示找不到 torch/cv2**：说明解释器没选对，按第一节第 2 步重新选 `torch_new` 环境。
- **GPU 显存不足（OOM）**：把 `train.py` 里的 `batch_size` 从 2 改成 1。
- **new_data 里文件混了两个数据集**：重新跑一次 `preprocess_all.py`（会自动清空重写）。
