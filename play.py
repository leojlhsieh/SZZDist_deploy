# %%
from datasets import load_from_disk  # Huggingface datasets  # pip install datasets
from tool.download_data import download_and_extract
from torchvision.transforms import v2
import torch

batch_size = 32
data_name = 'my_imagenette'
dataset_dir = str(download_and_extract(data_name))

# ds = load_from_disk(dataset_dir).with_format("torch")  # or .with_format("torch", device=device)
ds = load_from_disk(dataset_dir)
ds_train = ds['train'].select_columns(['image', 'feature_finetune_vit', 'label'])  # 'image', 'label', 'feature_finetune_vit', 'feature_pretrain_vit', 'logit_finetune_vit'
ds_test = ds['test'].select_columns(['image', 'feature_finetune_vit', 'label'])  # 'image', 'label', 'feature_finetune_vit', 'feature_pretrain_vit', 'logit_finetune_vit'


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


image_transform = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            v2.Resize(size=(300, 300)),  # it can have arbitrary number of leading batch dimensions
            v2.Grayscale(1),
        ]).to(device)


def transform(example):

    example["image"] = image_transform(example["image"].to(device))

    return example

ds_train = ds_train.map(transform, batched=True)
ds_test = ds_test.map(transform, batched=True)

i=-1
i
# %%
i += 1
ds_train[i]['image']

# %%
    if args.small_toy:
        ds_train = ds_train.select(range(args.batch_size*5))
        ds_test = ds_test.select(range(args.batch_size*5))
    
    def transforms(examples):
        examples["image"] = [v2.PILToTensor()(image) for image in examples["image"]]
        return examples
    ds_train.set_transform(transforms)
    ds_test.set_transform(transforms)

    train_dataloader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_cpu_worker,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=args.num_cpu_worker,
        pin_memory=True,
    )
    data_info = {
        'data_name': data_name,
        'batch_size': batch_size,
        'train_samples': len(ds_train),
        'test_samples': len(ds_test),
        'train_batches': len(train_dataloader),
        'test_batches': len(test_dataloader),
    }

    return train_dataloader, test_dataloader, data_info




























# %%
import torch
import torch.nn as nn
import torch.optim as optim
from model.leo_model_v20241015 import build_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bpm_color = 'gray'
bpm_mode = 'bpm'
bpm_depth = 4
bpm_width = 300
bpm_parallel = 1
model_feature = 'maxpool30-ReLU'


model_bpm, model_feature, model_classifier = build_model(bpm_color, bpm_mode, bpm_depth, bpm_width, bpm_parallel, model_feature, device=device)


# %%
def count_parameters(model):
    abc = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {abc:,} trainable parameters')
    return None

model = model_feature
model = model_classifier
model = model_bpm

for name, data in model.named_parameters():
    print(f'{name=}, {data.shape=}, {data.requires_grad=}, {data.numel()=}, {data.device=}')
    print('---')

# %%
import numpy as np
np.histogram



# %%

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example: input size 10, output size 1

    def forward(self, x):
        return self.fc(x)

# Create a model instance
model = SimpleModel()

# Example optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Assume we have trained the model here...

# %%
# get the prameters of the model
model_parameters = model.state_dict()
model_parameters

# %%
# only get trainable parameters
model_parameters = {k: v for k, v in model.state_dict().items() if v.requires_grad}
model_parameters

# %%
for i in model.parameters():
    print(i)

# %%

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for i in model.named_parameters():
    print(i)
    print('---')

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"



