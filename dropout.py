import torch
from torch import nn
import utils as d2l


def dropout_layer(X,dropout):
    #dropout在0和1之间
    assert 0 <= dropout <= 1
    #全丢，直接返回0
    if dropout == 1:
        return torch.zeros_like(X)
    #全不丢，返回X
    if dropout == 0:
        return X
    #保留大于dropout的值，带入公式，返回丢弃后的X
    mask = (torch.rand(X.shape)>dropout).float()
    return mask * X / (1.0-dropout)

num_inputs , num_outputs ,num_hiddens1 ,num_hiddens2 = 784 , 10 , 256 , 256

dropout_1 , dropout_2 = 0.2 , 0.5

#定义一个具有双隐藏层的模型
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training = True):
        super(Net,self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu = nn.ReLU()

    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape(-1,self.num_inputs)))
        if self.training == True:
            H1 = dropout_layer(H1,dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout_2)
        out = self.lin3(H2)
        return out

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

#net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256),nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout_1),nn.Linear(256, 256),nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout_2),nn.Linear(256, 10))

net.apply(init_weights)

num_epochs, lr, batch_size = 5, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, None,None,optimizer=trainer)