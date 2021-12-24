import torch

a1 = torch.rand(2, 31)
a2 = torch.rand(2, 31)
a3 = torch.rand(2, 31)

b1 = torch.rand(2)
b2 = torch.rand(2)
b3 = torch.rand(2)

a = torch.cat((a1, a2), 0)
print(a.shape)
