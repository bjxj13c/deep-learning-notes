# -*- coding: utf-8 -*-
"""
平移实验：同一张 7 右移若干像素，看第一层特征图是不是"跟着平移"。

关键：先在 preprocess 之后得到 28x28 张量，再平移。
     不能先平移原图再 preprocess —— 那样居中会把平移量抵消掉。
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from predict import preprocess, build_model

HERE    = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.abspath(os.path.join(HERE, "..", ".."))

model = build_model()
model.load_state_dict(torch.load(os.path.join(HERE, "mnist_cnn.pth"), map_location="cpu"))
model.eval()

SHIFTS = [0, 3, 6]


def shift_image(a, dy, dx):
    """把二维数组向下移 dy 像素、向右移 dx 像素，空出来的地方填 0。
       dy、dx 必须 >= 0。"""
    h, w = a.shape
    out = np.zeros_like(a)
    out[dy:, dx:] = a[:h - dy, :w - dx]
    return out


# ---------- 1. 造出三张平移图（唯一变量：位置） ----------
base = preprocess(os.path.join(HERE, "_debug", "7_28x28.png"))[0, 0].numpy()

shifted = []
print("=== 1. 三张平移图 ===")
for s in SHIFTS:
    b = shift_image(base, 0, s)
    shifted.append(b)
    ys, xs = np.where(b > 0.2)
    print("  右移 %d 像素 ->  行 %2d~%2d   列 %2d~%2d" % (s, ys.min(), ys.max(), xs.min(), xs.max()))

# ---------- 2. 取第一层特征图 ----------
feats = []
print()
print("=== 2. 第一层特征图 ===")
for s, b in zip(SHIFTS, shifted):
    t = torch.tensor(b, dtype=torch.float32).view(1, 1, 28, 28)
    with torch.no_grad():
        feats.append(model.features(t)[0].numpy())
        pred = int(torch.softmax(model(t), dim=1).argmax())
    print("  右移 %d 像素  ->  特征图 %s   整网预测 = %d" % (s, tuple(feats[-1].shape), pred))

# ---------- 3. 对齐验证：核心 ----------
# 若卷积层"找的是局部形状"，那么 把基准的特征图也右移 s 像素，就该 ≈ 右移 s 后的特征图。
# 注意：基准特征图右移 s 后，最右 s 列是补的 0，所以只比左边 (24-s) 列。
print()
print("=== 3. 对齐验证（平均绝对差，越小越像） ===")
print("  %-9s %10s %10s %8s" % ("filter", "对齐后", "未对齐", "差几倍"))
for s in [3, 6]:
    print("  --- 右移 %d 像素 ---" % s)
    for k in range(6):
        f_ref = feats[0][k]                     # 未平移的特征图
        f_mov = feats[SHIFTS.index(s)][k]       # 平移 s 后的特征图

        # 正确对齐：原图第 j 列的特征，平移后应出现在第 j+s 列。
        # 只比内部区域 —— 平移后右边界补了 0，边界附近本来就不可比（边界效应）。
        n = 24 - s
        aligned = np.abs(f_ref[:, :n] - f_mov[:, s:s + n]).mean()
        naive   = np.abs(f_ref[:, :n] - f_mov[:, :n]).mean()
        ratio   = naive / aligned if aligned > 1e-9 else float("inf")
        print("  %-9s %10.4f %10.4f %7.1fx" % ("filter%d" % k, aligned, naive, ratio))

# ---------- 4. 画图 ----------
fig, axes = plt.subplots(3, 7, figsize=(14, 6.5))

for r, s in enumerate(SHIFTS):
    axes[r, 0].imshow(shifted[r], cmap="gray", vmin=0, vmax=1)
    axes[r, 0].set_title("shift +%d px" % s, fontsize=9)

# 每列一个 filter，上下三行共用色标（才能公平比较）
for k in range(6):
    vmax = max(feats[r][k].max() for r in range(3))
    for r in range(3):
        axes[r, k + 1].imshow(feats[r][k], cmap="gray", vmin=0, vmax=vmax)
    axes[0, k + 1].set_title("filter %d" % k, fontsize=9)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("shift_feature_maps.png", dpi=120)
plt.show()
