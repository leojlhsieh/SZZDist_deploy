# %%
import torch

a = torch.randn(10, 10).cuda()

print(a.device, a.dtype, a.shape)

b = torch.tensor(3.1).cuda()

c = a[:b,:]

print(c.device, c.dtype, c.shape)


