import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features , labels = d2l.synthetic_data(true_w,true_b,1000)


def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)   #用TensorDataset对数据进行获取打包
    return data.DataLoader(dataset,batch_size,shuffle=is_train)         #用DataLoader抛出数据

batch_size = 10
data_iter = load_array((features,labels),batch_size)        #设置一个生成器


from torch import nn

#设置初始参数，设置模型，用Sequential容器包裹
net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.1)
net[0].bias.data.fill_(0)


#设置损失函数，SGD优化算法
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

#开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l = loss(net(x),y)
        trainer.zero_grad()      #梯度清零
        l.backward()
        trainer.step()           #更新模型
    l = loss(net(features),labels)
    print(f"epoch {epoch+1} , loss {l:f}")
