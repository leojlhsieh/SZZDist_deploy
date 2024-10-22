import torch
import torchvision
import torchvision.transforms.v2 as transforms

# random tensor with value between 0 and 1
image = torch.randn(2, 3, 4, 3000, 4000)
image *= 0.5
image += 0.5
print(image.mean(dim=(-2, -1)), image.std(dim=(-2, -1)))


ttt = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

image = ttt(image)
print(image.mean(dim=(-2, -1)), image.std(dim=(-2, -1)))
