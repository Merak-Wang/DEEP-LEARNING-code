import torch
from torch import nn
from d2l import torch as d2l

#二维互相关计算，X为输入，K为卷积核
def corr2d(X,K):
    #拿出K的行数和列数h，w
    h , w = K.shape
    #输出的高、宽为输入的高宽减核的宽度加一
    Y = torch.zeros((X.shape[0] - h + 1 , X.shape[1] - w + 1))
    #第一维度
    for i in range(Y.shape[0]):
        #第二维度
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x,self.weight) + self.bias


# X = torch.ones((6,8))
# X[:,2:6] = 0
# K = torch.tensor([[1.0,-1.0]])
#
# #边缘检测
# Y = corr2d(X,K)

# #学习上述卷积核
# #输入通道为1，输出通道为1
# conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
# X = X.reshape((1,1,6,8))
# #Y为卷积后输出的大小
# Y = Y.reshape((1,1,6,7))
#
# for i in range(10):
#     Y_hat = conv2d(X)
#     l = (Y_hat - Y)**2
#     conv2d.zero_grad()
#     l.sum().backward()
#     conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
#     if (i+1)%2 == 0:
#         print(f'batch {i+1}, loss {l.sum():.3f}')
#
# conv2d.weight.data.reshape((1,2))


#多输入多输出通道互关运算
def corr2d_multi_in(X,K):
    return sum(corr2d(x,k) for x,k in zip(X,K))

# X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
#                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

#corr2d_multi_in(X, K)


#不同卷积核输出在0维上堆叠起来
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

# K = torch.stack((K,K+1,K+2),0)

#print(K.shape)
#corr2d_multi_in_out(X,K)


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6