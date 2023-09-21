import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


#将图片转化成tensor
trans = transforms.ToTensor()
#下载数据用作训练和测试
mnist_train = torchvision.datasets.FashionMNIST(root = "../data",train = True ,transform=trans ,download=True)
mnist_test = torchvision.datasets.FashionMNIST(root = "../data",train = False ,transform=trans ,download=True)

batch_size = 256
#设置4个进程来读取数据
def get_dataloader_workers():
    return 4

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

timer = d2l.Timer()






