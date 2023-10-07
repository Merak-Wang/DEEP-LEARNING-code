import torch



# x=torch.arange(12)
#
# x=x.reshape(3,4)
#
# x=torch.tensor([1.0,2,4,8])
#
# y=torch.tensor([2,2,2,2])

# x = torch.arange(12,dtype=torch.float32).reshape((3,4))
#
# y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
#
# a=torch.arange(3).reshape((3,1))
#
# b=torch.arange(2).reshape((1,2))
#
# A = x.numpy()
#
# B=torch.tensor(A)

import os
#
# os.makedirs(os.path.join('..','data'),exist_ok=True)
# data_file = os.path.join('..','data','house_tiny.csv')
# with open(data_file,'w') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
#
import pandas as pd
#
# data = pd.read_csv(data_file)


# inputs , outputs = data.iloc[:,0:2] , data.iloc[:,2]
#
# inputs = inputs.fillna(inputs.mean())
#
# inputs = pd.get_dummies(inputs,dummy_na=True)
#
# x , y =torch.tensor(inputs.values) , torch.tensor(outputs.values)

# x = torch.arange(24,dtype=torch.float32).reshape(2,3,4)

# x = torch.arange(4.0)
#
# x.requires_grad_(True)
#
# y = 2 * torch.dot(x,x)
# matrix =[[2,1,3],[6,5,4],[7,8,9]]
#
# n = len(matrix[0])
# for i in range(n-2,-1,-1):
#     for j in range(n):
#         if j == 0:
#             matrix[i][j] += min(matrix[i+1][j] , matrix[i+1][j+1])
#         elif j == n-1:
#             matrix[i][j] += min(matrix[i+1][j] , matrix[i+1][j-1])
#         else:
#             matrix[i][j] += min(matrix[i+1][j-1] , matrix[i+1][j] , matrix[i+1][j+1])
#
# print(matrix)