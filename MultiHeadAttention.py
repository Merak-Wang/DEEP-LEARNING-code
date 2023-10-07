import math
import torch
from torch import nn
from d2l import torch as d2l

class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,quary_size,value_size,num_hiddens,
                 num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        #多头注意力，头的数量
        self.num_heads = num_heads
        #点积注意力，无可学习的参数
        self.attention = d2l.DotProductAttention(dropout)
        #可学习参数
        self.W_q = nn.Linear(quary_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)

    def forward(self,quries,keys,valus,vaild_lens):
        quries = transpose_qkv(self.W_q(quries),self.num_heads)
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        valus = transpose_qkv(self.W_v(valus),self.num_heads)

        if vaild_lens is not None:
            vaild_lens = torch.repeat_interleave(vaild_lens,
                                                 repeats=self.num_heads,
                                                 dim = 0)

        output = self.attention(quries,keys,valus,vaild_lens)
        output_cancat = transpose_output(output,self.num_heads)
        return self.W_o(output_cancat)

def transpose_qkv(X,num_heads):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X,num_heads):
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

# num_hiddens, num_heads = 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
#                                num_hiddens, num_heads, 0.5)
# attention.eval()
#
# batch_size, num_queries = 2, 4
# num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
# X = torch.ones((batch_size, num_queries, num_hiddens))
# Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
#attention(X, Y, Y, valid_lens).shape

