# -*- coding: utf-8 -*-
"""
数据增强对照实验：训练时加随机平移，能不能提升平移鲁棒性？
   原 CNN   —— 不加增强（直接加载 mnist_cnn.pth）
   增强 CNN —— 训练时随机平移（本脚本现场训练，存 mnist_cnn_aug.pth）
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from model import MyCNN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")
EPOCHS = 30
SHIFTS = [0, 1, 2, 3, 5]

tf = transforms.Compose([transforms.ToTensor()])            # 测试集用（干净）
tf_aug = transforms.Compose([                               # 训练集用（带随机平移）
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

train_set = MNIST(DATA_ROOT, train=True,  transform=tf_aug, download=False)   # ← 增强版
test_set  = MNIST(DATA_ROOT, train=False, transform=tf,     download=False)   # ← 干净版

train_loader = DataLoader(train_set, batch_size=64,   shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

loss_fn = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, loss_fn, optimizer):
    """训练一轮，返回 (平均损失, 训练准确率)"""
    model.train()
    total_loss = 0.0
    correct = total = 0
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, shift=0):
    """评估准确率。shift>0 时把图整体右移，左边补黑。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            if shift > 0:                                   # 平移只在评估时做
                out = torch.zeros_like(x)
                out[:, :, :, shift:] = x[:, :, :, :-shift]
                x = out
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ==================== 1. 训练"增强版" ====================
print("=== 训练增强版 CNN（60000 张 + 随机平移，%d 轮）===" % EPOCHS)
model_aug = MyCNN()
optimizer = torch.optim.SGD(model_aug.parameters(), lr=0.1)
for epoch in range(EPOCHS):
    loss, acc = train_one_epoch(model_aug, train_loader, loss_fn, optimizer)
    print("第%2d轮 损失%.4f  训练准确度%.4f" % (epoch + 1, loss, acc))
torch.save(model_aug.state_dict(), os.path.join(HERE, "mnist_cnn_aug.pth"))
print("已保存 mnist_cnn_aug.pth")

# ==================== 2. 加载"原版" ====================
model_old = MyCNN()
model_old.load_state_dict(torch.load(os.path.join(HERE, "mnist_cnn.pth"), map_location="cpu"))
model_old.eval()

# ==================== 3. 对比平移鲁棒性 ====================
print()
print("=== 平移鲁棒性对比 ===")
print("平移量     原CNN     增强CNN     提升")
print("-" * 44)
for s in SHIFTS:
    a = evaluate(model_old, test_loader, s) * 100
    b = evaluate(model_aug, test_loader, s) * 100
    print(" %d 像素    %6.2f%%   %6.2f%%    %+.2f" % (s, a, b, b - a))
