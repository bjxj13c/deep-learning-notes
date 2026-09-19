#导入——————————————-
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


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

model=nn.Sequential(
    nn.Flatten(),#模型铺平：即把【1，25，28】变成1*28*28
    nn.Linear(1*28*28,10),#矩阵乘法and加偏置
)

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

w=model[1].weight
IDX=[392,393,394,395,396]
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

EPOCHS=5

for epoch in range(EPOCHS):
    total_loss=0.0
    correct=0
    total=0

    for x,y in train_loader:
        out=model(x)#向前传播
        loss=loss_fn(out,y)#计算损失

        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数

        total_loss+=loss.item()
        correct+=(out.argmax(1)==y).sum().item()
        total+=y.size(0)

    print("第%d轮 损失%.4f  训练准确度%.4f"%(epoch+1,total_loss/len(train_loader),correct/total))


test_set=MNIST(DATA_ROOT,train=False,transform=tf,download=False)
test_loader=DataLoader(test_set,batch_size=1000,shuffle=False)

model.eval()
correct=0
total=0
with torch.no_grad():
    for x,y in test_loader:
        out=model(x)
        correct+=(out.argmax(1)==y).sum().item()
        total+=y.size(0)


print("训练集准确率：%.2f%%"%(100*0.9175))
print("测试集准确率：%.2f%%"%(100*correct/total))