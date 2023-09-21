from d2l import torch as d2l
import random
import torch


#构造一个人造数据集：w线性模型参数w=[2,-3.4]、b=4.2和噪声项
#y = xw + b +噪声
def synthetic_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape)
    return x , y.reshape((-1,1)) #以列向量返回y

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features , labels = synthetic_data(true_w,true_b,1000)


#实现一个函数，每次读取小批量数据
def data_iter(batch_size,features,labels):
    num_example = len(features)

    # 将下标打乱
    indices = list(range(num_example))
    random.shuffle(indices)

    #返回生成器
    for i in range(0,num_example,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_example)])
        yield features[batch_indices] , labels[batch_indices]


#读取batch_size批量的数据
# batch_size = 10
# for x,y in data_iter(batch_size,features,labels):
#     print(x,'\n',y)
#     break

#初始化模型参数：通过正态分布
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

#定义模型
def linreg(x,w,b):
    return torch.matmul(x,w) + b  #线性回归

#定义损失函数：均方误差，y_hat为预测值，y为真实值
def squared_loss(y_hat,y):
    return ((y_hat-y.reshape(y_hat.shape))**2)/2   #防止y_hat和y形状不一致

#定义优化函数：随机梯度下降算法
def sgd(params,lr,batch_size):                  #params为数据（包含w，x），lr为学习率
    with torch.no_grad():                       #接下来不需要计算梯度
        for param in params:
            param -= lr*param.grad / batch_size
            param.grad.zero_()                  #梯度清零，每个batch不要累积梯度

#设置超参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

#训练过程
for epoch in range(num_epochs):                             #扫三遍数据
    for x,y in data_iter(batch_size,features,labels):       #取出每个batch
        l = loss(net(x,w,b),y)                              #损失为向量
        l.sum().backward()                                  #算梯度
        sgd([w,b],lr,batch_size)                            #算完梯度后用SGD更新w，b的参数

    #每扫一次，评估，不需要计算梯度
    with torch.no_grad():
        train_1 = loss(net(features,w,b),labels)            #预测值和真实值算损失
        print(f"epoch {epoch+1},loss {float(train_1.mean()):.6f}")


#估计误差
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
