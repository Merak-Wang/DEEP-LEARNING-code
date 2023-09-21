import torch
from d2l import torch as d2l

#设计20个数据训练，训练数越小，模型越复杂，过拟合越容易发生
n_train , n_test , num_inputs , batch_size = 20 , 100 , 200 , 5
true_w , true_b = torch.zeros((num_inputs,1))*0.01 , 0.05
train_data = d2l.synthetic_data(true_w,true_b,n_train)
train_iter = d2l.load_array(train_data,batch_size)
test_data = d2l.synthetic_data(true_w,true_b,n_test)
test_iter = d2l.load_array(test_data,batch_size,is_train=False)

#初始化
def init_params():
    w = torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

#定义L2范数惩罚
def L2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def L1_penalty(w):
    return torch.sum(torch.abs(w))

def train(lambd):
    w,b = init_params()
    net , loss = lambda X: torch.matmul(X,w)+b , d2l.squared_loss
    num_epochs , lr = 100 , 0.003
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X),y) + lambd * L1_penalty(w)  #添加L2正则
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        with torch.no_grad():
            train_1 = d2l.evaluate_loss(net,train_iter,loss)
            test_1 = d2l.evaluate_loss(net, test_iter, loss)
            print(f"epoch {epoch + 1},train acc {float(train_1):.6f}, test acc {float(test_1):.6f}")
    print('w的L1范数是：', torch.norm(w).item())

train(lambd=3)


