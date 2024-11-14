# %%
import torch

arr = torch.rand(5000, 3000)

arr



arr = arr.to("cuda")


print(123)
from tqdm import tqdm

for i in tqdm(range(1000)):
    arr = torch.fft.fft2(arr)





# %%
