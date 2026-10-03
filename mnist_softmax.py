#导入——————————————-
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model import MyCNN

#数据集位置
DATA_ROOT="./data"

#预处理（转换）
tf=transforms.Compose([transforms.ToTensor()])

#拿取数据
train_set=MNIST(DATA_ROOT,train=True,transform=tf,download=True)

print("训练集张数：",len(train_set))

#batch_size，一次拿取的数量
train_loader=DataLoader(train_set,batch_size=64,shuffle=True)

x,y=next(iter(train_loader))

print("x的形状：",x.shape)
print("y的形状：",y.shape)
print("y的前10标签：",y[:10].tolist())
print("像素值范围：%.2f~%.2f"%(x.min(),x.max()))

model=MyCNN()

def train_one_epoch(model,loader,loss_fn,optimizer):
    """训练一轮返回平均损失和训练准确率"""
    model.train()

    total_loss=0.0
    correct=0
    total=0

    for x,y in loader:
        out=model(x)#前向传播
        loss=loss_fn(out,y)#算损失

        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数

        total_loss+=loss.item()
        correct+=(out.argmax(1)==y).sum().item()
        total+=y.size(0)

    return total_loss/len(loader),correct/total

def evaluate(model,loader):
    """给定的数据集上评估，返回准确率"""
    model.eval()

    total=0
    correct=0
    with torch.no_grad():#只进行推理，不记录梯度
        for x,y in loader:
            correct+=(model(x).argmax(1)==y).sum().item()
            total+=y.size(0)

    return correct/total

print(model)

n_param=sum(p.numel() for p in model.parameters())
print("参数量：",n_param)

out=model(x)
print("输出的形状：",out.shape)
print("第一张图的10个得分",out[0].tolist())


#算损失
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(out,y)

print("损失：",loss.item())

optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

#验证梯度下降
w=model.fc1.weight
IDX=[0,1,2,3,4]
w_before=w[0][IDX].clone()

out=model(x)
loss=loss_fn(out,y)
print("损失：",loss.item())

optimizer.zero_grad()
loss.backward()
g=w.grad[0][IDX]
print("梯度：",g.tolist())

optimizer.step()

print("更新前：",w_before.tolist())
print("更新后：",w[0][IDX].tolist())
print("变化量：",(w[0][IDX]-w_before).tolist())
print("校验-lr*梯度：",(-0.1*g).tolist())

EPOCHS=30#轮数

for epoch in range(EPOCHS):
    train_loss,train_acc=train_one_epoch(model,train_loader,loss_fn,optimizer)
    print("第%d轮损失：%.4f 训练准确度：%.4f"%(epoch+1,train_loss,train_acc))

test_set=MNIST(DATA_ROOT,train=False,transform=tf,download=False)
test_loader=DataLoader(test_set,batch_size=1000,shuffle=False)

test_acc=evaluate(model,test_loader)

print("训练集准确率：%.2f"%(100*train_acc))
print("测试集准确率：%.2f"%(100*test_acc))



#保存模型

torch.save(model.state_dict(),"mnist_cnn.pth")
print("模型保存至mnist_cnn.pth")