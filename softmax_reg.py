import torch
from torch import nn
from torch.nn import init
import d2lzh_pytorch as d2l

#下载MNIST数据进行测试，将图片转化成tensor
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_input = 784
num_output = 10
num_epochs = 5


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

#定义模型
net = LinearNet(num_input, num_output)
#初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
#使用交叉熵损失函数
loss = nn.CrossEntropyLoss()
#随机梯度下降优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
#开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)