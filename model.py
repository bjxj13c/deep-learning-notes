"""模型结构定义"""

import torch
import torch.nn.functional as F
from torch import nn

class MyCNN(nn.Module):
    """手写数字分类的CNN"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size=5)
        self.conv2 = nn.Conv2d(6,16,kernel_size=5)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120, 10)

    def features(self,x):
        """第一层卷积加relu"""
        return F.relu(self.conv1(x))

    def forward(self,x):
        """前向传播"""
        x = F.relu(self.conv1(x))#卷积+relu
        x = F.max_pool2d(x,2)#池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x=torch.flatten(x,1)#铺平
        x = F.relu(self.fc1(x))#全连接+relu
        x = self.fc2(x)
        return x
if __name__ == "__main__":
     m = MyCNN()
     print("参数量:", sum(p.numel() for p in m.parameters()))
     print("features 输出:", m.features(torch.randn(1, 1, 28, 28)).shape)
     print("forward  输出:", m(torch.randn(1, 1, 28, 28)).shape)
