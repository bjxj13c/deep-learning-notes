# -*- coding: utf-8 -*-
"""
实验：图片平移对 MLP 和 CNN 的影响

【为什么做这个实验】
    我训练了一个测试集 98.05% 的 MLP（全连接），
    但它对两张"人眼看着完全正常"的手写 7 给出了错误答案
    （预测 2，且 7 的概率只有 0.002）。

    换成 CNN 后，同样这两张图全部答对（85% 和 99%）。

【推断的原因】
    MLP 学的是「固定位置的像素模板」——  权重 w[0][392] 只对"第 392 号位置"敏感。
    CNN  学的是「局部形状」—— 同一个卷积核滑遍全图，换个位置照样能检测到。

    所以：MLP 没有「平移不变性」，CNN 有。

【这个实验怎么验证】
    把 MNIST 测试图整体向右平移 k 个像素，测两个模型的准确率。
    如果 MLP 掉得厉害、CNN 几乎不动 → 推断成立。

【用法】
    python exp_translation.py

【前提】
    mnist_cnn.pth 已存在（跑一次 mnist_softmax.py）
    mnist_mlp.pth 不存在时会自动训练一个（约 1-2 分钟）
"""
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CNN_PATH = os.path.join(HERE, "mnist_cnn.pth")
MLP_PATH = os.path.join(HERE, "mnist_mlp.pth")

SHIFTS = [0, 1, 2, 3, 5]          # 向右平移几个像素
EPOCHS = 30
LR = 0.1


# ────────────────────────────────────────────────────────────
# 两个模型的结构（必须和训练时一致）
# ────────────────────────────────────────────────────────────
def build_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        nn.ReLU(),
        nn.Linear(120, 10),
    )


def train(model, loader, tag):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("      %s 训练完成（%.0f 秒）" % (tag, time.time() - t0))
    return model


def evaluate(model, loader, shift):
    """测准确率。shift>0 时把图整体向右平移，左边补黑（0）。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            if shift > 0:
                out = torch.zeros_like(x)          # MNIST 背景是 0（黑）
                out[:, :, :, shift:] = x[:, :, :, :-shift]
                x = out
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# ────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CNN_PATH):
        raise SystemExit("找不到 %s\n请先跑一遍 mnist_softmax.py" % CNN_PATH)

    tf = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST(DATA, train=True, download=False, transform=tf)
    test_set = MNIST(DATA, train=False, download=False, transform=tf)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    print("=" * 62)
    print("准备两个模型")
    print("=" * 62)

    # ── CNN：直接加载 ──
    cnn = build_cnn()
    cnn.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
    cnn.eval()
    print("      CNN  已加载：", CNN_PATH)

    # ── MLP：有就加载，没有就训练 ──
    mlp = build_mlp()
    if os.path.exists(MLP_PATH):
        mlp.load_state_dict(torch.load(MLP_PATH, map_location="cpu"))
        print("      MLP  已加载：", MLP_PATH)
    else:
        print("      MLP  没找到，开始训练（%d 轮）…" % EPOCHS)
        train(mlp, train_loader, "MLP")
        torch.save(mlp.state_dict(), MLP_PATH)
        print("      MLP  已保存：", MLP_PATH)
    mlp.eval()

    # ── 逐档平移测准确率 ──
    print()
    print("=" * 62)
    print("把测试图整体【向右平移】k 个像素，看准确率怎么变")
    print("=" * 62)
    print("  平移量      MLP        CNN      差距")
    print("-" * 62)

    res = {"mlp": [], "cnn": []}
    for k in SHIFTS:
        a = evaluate(mlp, test_loader, k)
        b = evaluate(cnn, test_loader, k)
        res["mlp"].append(a)
        res["cnn"].append(b)
        print("  %2d 像素    %6.2f%%    %6.2f%%    %+5.2f"
              % (k, a, b, b - a))
    print("-" * 62)
    print("  MLP 总共掉了 %.2f 个百分点" % (res["mlp"][0] - res["mlp"][-1]))
    print("  CNN 总共掉了 %.2f 个百分点" % (res["cnn"][0] - res["cnn"][-1]))

    # ── 画图 ──
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(SHIFTS, res["mlp"], "o-", color="#d64545", linewidth=2,
            markersize=7, label="MLP（全连接，101770 参数）")
    ax.plot(SHIFTS, res["cnn"], "s-", color="#0e9f6e", linewidth=2,
            markersize=7, label="CNN（卷积，34622 参数）")
    ax.set_xlabel("图片向右平移的像素数")
    ax.set_ylabel("测试集准确率 (%)")
    ax.set_title("平移对两个模型的影响")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for k, a, b in zip(SHIFTS, res["mlp"], res["cnn"]):
        ax.annotate("%.1f" % a, (k, a), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="#d64545")
        ax.annotate("%.1f" % b, (k, b), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9, color="#0e9f6e")
    out = os.path.join(HERE, "平移实验.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print()
    print("曲线图已保存：", out)

    # ── 存一份文字结果 ──
    txt = os.path.join(HERE, "平移实验结果.md")
    lines = [
        "# 平移实验：MLP vs CNN",
        "",
        "把 MNIST 测试图整体向右平移 k 个像素，测两个模型的准确率。",
        "",
        "| 平移量 | MLP | CNN | 差距 |",
        "|---|---|---|---|",
    ]
    for k, a, b in zip(SHIFTS, res["mlp"], res["cnn"]):
        lines.append("| %d 像素 | %.2f%% | %.2f%% | %+.2f |" % (k, a, b, b - a))
    lines += [
        "",
        "**MLP 总共掉了 %.2f 个百分点；CNN 总共掉了 %.2f 个百分点。**"
        % (res["mlp"][0] - res["mlp"][-1], res["cnn"][0] - res["cnn"][-1]),
        "",
        "结论：CNN 的平移鲁棒性明显更强 —— 这来自卷积的「权值共享」，",
        "同一个卷积核在图上任何位置都能用。",
    ]
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("文字结果已保存：", txt)


if __name__ == "__main__":
    main()
