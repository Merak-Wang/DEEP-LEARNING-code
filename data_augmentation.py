import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
#如果没有图片，通过以下代码导入
#!curl https://zh-v2.d2l.ai/d2l-zh.zip -o d2l-zh.zip
img = d2l.Image.open('D:/PyCharm Community Edition 2020.1.3/WMF_Projects/Py_Projects/d2l-zh/pytorch/img/cat1.jpg')
d2l.plt.imshow(img)

def apply (img,aug,num_rows=2,num_cols=4,scale=1.5):
    Y = [aug(img) for _ in range(num_rows*num_cols)]
    d2l.show_images(Y,num_rows,num_cols,scale=scale)

#水平翻转
#apply(img,torchvision.transforms.RandomHorizontalFlip())
#上下翻转
#apply(img,torchvision.transforms.RandomVerticalFlip())
#随即裁剪
#apply(img,torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2)))
#随机亮度
#apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0))
#改变色调
#apply(img,torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
#随即更改
#apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5))

apply(img,torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                          torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
                                          torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))]))

# all_image = torchvision.datasets.CIFAR10(train=True,root="../data",download=True)
#d2l.show_images([all_image[i][0] for i in range(32)],4,8,scale=0.8)

# train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
#                                              torchvision.transforms.ToTensor()])
# test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#
# def load_cifar10(is_train,augs,batch_size):
#     dataset = torchvision.datasets.CIFAR10(root="../data",train=is_train,transform=augs,download=True)
#     dataloder = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
#                                         num_workers=4)
#     return dataloder






