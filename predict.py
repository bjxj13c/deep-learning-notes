# -*- coding: utf-8 -*-
"""
用训练好的模型预测手写数字。

用法：
     python predict.py --test           # 从 MNIST 测试集抽一张验证流程
     python predict.py 图片路径          # 预测你自己的图片

前提：先跑 mnist_softmax.py 生成 mnist_cnn.pth
"""

import os
import sys

import numpy as np
import torch
from torch import nn
from PIL import Image
from scipy import ndimage

HERE=os.path.dirname(os.path.abspath(__file__))
WEIGHTS=os.path.join(HERE,"mnist_cnn.pth")

def build_model():
    # ★ 必须和 mnist_softmax.py 里的 model 定义【一字不差】，
    #   否则 load_state_dict 会因为参数名对不上而报错。
    #   （模型结构一改，这里就要同步改 —— 这就是"结构散在两个文件里"的痛点。）
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


def load_model():
    if not os.path.exists(WEIGHTS):
        raise SystemExit("找不到%s\n"%WEIGHTS)
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS,map_location="cpu"))
    model.eval()
    print("模型已加载；",WEIGHTS)
    return model


def preprocess(path):
    """把图片变成（1，1，28，28）张量"""
    a=np.array(Image.open(path).convert("L"))

    if a.mean()>127:
        a=255-a

    mask=a>50
    labeled,n=ndimage.label(mask)
    if n==0:
        raise SystemExit("找不到笔画")
    if n>1:
        sizes = ndimage.sum(mask,labeled,range(1,n+1))
        mask=(labeled==int (np.argmax(sizes))+1)
    a=np.where(mask,a,0).astype(np.uint8)

    ys,xs=np.where(a>50)
    a=a[ys.min():ys.max() + 1,xs.min():xs.max() + 1]

    h,w=a.shape
    s=20.0/max(h,w)
    nh,nw=max(1,round(h*s)),max(1,round(w*s))
    a=np.array(Image.fromarray(a).resize((nw,nh),Image.LANCZOS))

    canvas = np.zeros((28, 28), dtype=np.uint8)  # 贴到 28x28 正中央
    t, l = (28 - nh) // 2, (28 - nw) // 2
    canvas[t:t + nh, l:l + nw] = a

    return torch.tensor(canvas,dtype=torch.float32).view(1,1,28,28)/255.0

def predict(model, t):
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0]
    return int(probs.argmax()), probs

if __name__ =="__main__":
    model=load_model()

    if len(sys.argv) < 2:
        print("用法: python predict.py --test  或  python predict.py 图片路径")
        sys.exit(1)

    if sys.argv[1] == "--test":
          # 从 MNIST 测试集抽一张，存成 PNG 再走完整流程
          from torchvision.datasets import MNIST
          ds = MNIST(os.path.join(HERE, "data"), train=False, download=False,
                     transform=None)
          img, label = ds[0]
          out_png = os.path.join(HERE, "sample.png")
          img.save(out_png)
          print("从测试集抽了一张：真实标签 = %d，已存为 sample.png" % label)
          path = out_png
    else:
          path = sys.argv[1]
          label = None


    t = preprocess(path)
    pred, probs = predict(model, t)

    print()
    print("=" * 52)
    print("预测结果： %d" % pred)
    if label is not None:
        print("真实标签： %d   %s" % (label, "✅ 对了" if pred == label else "❌ 错了"))
    print("=" * 52)
    for i, p in enumerate(probs.tolist()):
        bar = "#" * int(p * 40)
        mark = "  <== 猜的是这个" if i == pred else ""
        print("  %d   %.4f  %s%s" % (i, p, bar, mark))