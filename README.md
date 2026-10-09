# U-Net Retina Blood Vessel Segmentation (PyTorch)

Semantic segmentation of retinal blood vessels with **U-Net** in PyTorch, supporting **4 datasets** out of the box: **DRIVE, CHASE_DB1, HRF and FIVES**.

This project is a heavily reworked and extended version of
[nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch](https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch),
with multi-dataset support, high-resolution patch training, sliding-window inference and fully commented code (Chinese comments).

## Features

- **Multi-dataset support**: DRIVE / CHASE_DB1 / HRF / FIVES — switch with one CLI argument
- **High-resolution patch training**: keep original resolution (HRF 3504x2336, FIVES 2048x2048) instead of losing detail by downscaling
- **Sliding-window inference**: stitch full-size predictions with 50% overlapping windows
- **Portable paths**: all scripts use absolute paths based on their own location — run from anywhere, no working-directory dependency
- **Fully commented code** (Chinese, line-by-line) — beginner friendly
- **Fixed bugs** from the original repo: removed deprecated PyTorch `ReduceLROnPlateau(verbose=)` argument; fixed DRIVE file-name pairing

## Results (reproduced on RTX 3070 Ti Laptop, 8GB)

| Dataset | Resolution | Test size | Jaccard | F1 | Recall | Precision | Acc |
|---|---|---|---|---|---|---|---|
| DRIVE | 584x565 (resized 512) | 20 | **0.6155** | 0.7613 | 0.7254 | 0.8096 | 0.9606 |
| HRF | 3504x2336 (patch) | 14 | **0.6690** | 0.7996 | 0.7785 | 0.8285 | 0.9686 |
| FIVES | 2048x2048 (patch) | 200 | **0.7268** | 0.8291 | 0.8058 | 0.8749 | 0.9809 |

## Quick Start

```bash
# Environment: Python 3.11 + PyTorch (CUDA recommended)
pip install -r requirements.txt

# 1) DRIVE (low-res, direct resize) — or: chase / hrf
python preprocess_all.py --dataset drive
python train.py
python test.py

# 2) HRF / FIVES (high-res, patch training — recommended)
python preprocess_hrf_full.py      # or preprocess_fives.py
python train_hrf_patch.py          # or train_fives_patch.py
python test_hrf_patch.py           # or test_fives_patch.py
```

Each dataset must be placed locally (data is not distributed in this repo):

| Dataset | Path used by scripts |
|---|---|
| DRIVE | `D:\eyedata\彩色眼底图数据库\DRIVE\DRIVE` |
| CHASE_DB1 | `D:\eyedata\CHASE_DB1` |
| HRF | `D:\eyedata\HRF` |
| FIVES | `D:\eyedata\FIVES` |

To use your own paths, edit the `SRC` variable at the top of the corresponding `preprocess_*.py`.

## Project Structure

```
├── preprocess_all.py          # generic preprocessing: --dataset drive|chase|hrf
├── preprocess_drive.py        # DRIVE-only legacy script
├── preprocess_hrf_full.py     # HRF full-resolution split (31/14)
├── preprocess_fives.py        # FIVES full-resolution split (600/200, official)
├── train.py / test.py         # standard training & evaluation (DRIVE-scale)
├── train_hrf_patch.py         # HRF patch training (online random crops)
├── test_hrf_patch.py          # HRF sliding-window inference
├── train_fives_patch.py       # FIVES patch training
├── test_fives_patch.py        # FIVES sliding-window inference
├── model.py                   # U-Net definition
├── data.py                    # dataset loader
├── loss.py                    # Dice + BCE loss
├── utils.py                   # seeding / helpers
├── data_aug.py                # optional augmentation (requires albumentations)
└── requirements.txt
```

Chinese usage guide (PyCharm): see `使用说明-PyCharm.md`.

## Datasets

- **DRIVE**: [Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/) — 40 images, official train/test split
- **CHASE_DB1**: 28 retinal images, no official split (script uses 70/30 with fixed seed)
- **HRF**: [High-Resolution Fundus](https://www5.cs.fau.de/research/data/fundus-images/) — 45 images, 3504x2336
- **FIVES**: [Finland Retinal Vessel Segmentation](https://github.com/AbdullahFalcon/FIVES) — 800 images, 2048x2048, official 600/200 split

Please follow each dataset's license/terms for academic use.

## Acknowledgements

- U-Net architecture: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Original code base: [nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch](https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch)

## License

[MIT](LICENSE)
