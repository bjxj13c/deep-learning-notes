# -*- coding: utf-8 -*-
"""
 可视化 CNN 第一层卷积的 6 张特征图。
 三张 7 并排对比：桌面 7.png、桌面 7.1.png、MNIST 真 7
"""

import os

import torch

import matplotlib.pyplot as plt

from predict import preprocess,build_model

HERE   = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.abspath(os.path.join(HERE,"..",".."))

IMAGES=[
    ("Desktop 7.png", os.path.join(DESKTOP, "7.png")),
    ("Desktop 7.1.png", os.path.join(DESKTOP, "7.1.png")),
    ("MNIST real 7", os.path.join(HERE, "_debug", "mnist7.png")),
]

model=build_model()
model.load_state_dict(torch.load(os.path.join(HERE,"mnist_cnn.pth"),map_location="cpu"))
model.eval()


print()
print("权重加载，参数量=",sum(p.numel() for p in model.parameters()))


print("脚本目录：",HERE)
print("桌面目录:", DESKTOP)
print()
for name, path in IMAGES:
    print("%-16s %s  %s" % (name, path, "OK" if os.path.exists(path) else "找不到"))


tensors=[]
for name, path in IMAGES:
    t=preprocess(path)
    tensors.append(t)

    with torch.no_grad():
        probs=torch.softmax(model(t),dim=1)[0]
    pred=int(probs.argmax())

    print("%-16s 形状=%s  预测=%d  置信度=%.4f" % (name, tuple(t.shape), pred, probs[pred]))


print()
print("模型结构：")
for i,layer in enumerate(model):
    print("model[%d]=%s"%(i,layer))

feats=[]
for (name, path), t in zip(IMAGES, tensors):
      with torch.no_grad():
          f = model[:2](t)
      feats.append(f)
      print("%-16s %s -> %s" % (name, tuple(t.shape), tuple(f.shape)))

print("切片是复制还是共享？", model[:2][0] is model[0])

fig,axes=plt.subplots(3,7,figsize=(14,6.5))

# 左起第 0 列：三张原图（统一色标 0~1，才能公平比较笔画粗细/亮度）
for r, (name, _) in enumerate(IMAGES):
    axes[r, 0].imshow(tensors[r][0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[r, 0].set_title(name, fontsize=9)

# 第 1~6 列：每列是一个 filter，上下三行共用同一个色标
# 注意循环顺序：外层 k(filter)、内层 r(图片)，这样同一个 filter 的三张图合用一把尺子
for k in range(6):
    vmax = max(feats[r][0, k].max().item() for r in range(3))
    for r in range(3):
        axes[r, k + 1].imshow(feats[r][0, k].numpy(), cmap="gray", vmin=0, vmax=vmax)
    axes[0, k + 1].set_title("filter %d" % k, fontsize=9)

for ax in axes.flat:
      ax.set_xticks([])
      ax.set_yticks([])

plt.tight_layout()
plt.savefig("feature_maps.png", dpi=120)
plt.show()