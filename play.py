# %% 
from tool.build_dataloader import build_dataloader
import logging
import torch
from torchvision.transforms import v2


logging.basicConfig(level=logging.DEBUG)
data_name = 'my_cifar10'  # 'my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'
batch_size = 128
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
small_toy = 7
train_dataloader, test_dataloader, data_info = build_dataloader(data_name, batch_size, device, small_toy)

image_transform = v2.Compose([
    # v2.PILToTensor(),
    v2.Grayscale(1),
    v2.Resize(size=(300, 300)),  # it can have arbitrary number of leading batch dimensions
    v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
])

# %%
for batch, data in enumerate(train_dataloader):
    print(f"image, {data['image'].shape}, {data['image'].dtype}, {data['image'].device}")
    print(f"label, {data['label'].shape}, {data['label'].dtype}, {data['label'].device}")
    print(f"feature_finetune_vi, {data['feature_finetune_vit'].shape}, {data['feature_finetune_vit'].dtype}, {data['feature_finetune_vit'].device}")
    
    image = image_transform(data['image'])
    feature = data['feature_finetune_vit']
    label = data['label']
    
    break

# %%
