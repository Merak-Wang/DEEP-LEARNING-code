import torch
from torch import nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#ReLU:max(x,0)
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

#通过view函数将每张原始图像改成长度为num_inputs的向量
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


#引入数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#设置一个隐藏层
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

#交叉熵损失函数
loss = nn.CrossEntropyLoss()
optimizer  = torch.optim.SGD(params, lr=0.1)

#开始训练
num_epochs, lr = 10, 100
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr,optimizer)
