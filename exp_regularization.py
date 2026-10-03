# -*- coding: utf-8 -*-
"""
正则化实验：故意造过拟合 → 逐个加正则化手段，看差距缩小多少
对照组（病态基线）：只用 500 张训练图 + 训 100 轮
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset      # ← Subset 从这里来
from torchvision import transforms
from torchvision.datasets import MNIST
from model import MyCNN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")
tf = transforms.Compose([transforms.ToTensor()])
tf_aug = transforms.Compose([transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
                             transforms.ToTensor(),
                             ])


train_set = MNIST(DATA_ROOT, train=True,  transform=tf_aug, download=False)
test_set  = MNIST(DATA_ROOT, train=False, transform=tf, download=False)

small_set    = Subset(train_set, range(500))        # 只取前 500 张
small_loader = DataLoader(small_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

loss_fn = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, loss_fn, optimizer):
    """训练一轮，返回 (平均损失, 训练准确率)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
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

def evaluate(model, loader):
    """评估，返回准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ==================== 病态基线 ====================
model = MyCNN(dropout=0.5)                                     # ⚠️ 必须新建
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,weight_decay=1e-2)
print("=== 病态基线：500 张图 + 100 轮 ===")
print("轮数      训练集     测试集     差距")
for epoch in range(200):
    tr_loss, tr_acc = train_one_epoch(model, small_loader, loss_fn, optimizer)
    te_acc = evaluate(model, test_loader)
    if (epoch + 1) % 10 == 0:                       # 每 10 轮打印一次
        print("%3d      %.4f     %.4f    %+.4f" % (epoch+1, tr_acc, te_acc, tr_acc - te_acc))
