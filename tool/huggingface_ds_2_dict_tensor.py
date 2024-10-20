# %%
from pprint import pprint
from datasets import load_from_disk  # Huggingface datasets  # pip install datasets
import torch

if __name__ == '__main__':
    # Add parent directory to sys.path temporarily, so this file can be run directly as main
    from pathlib import Path
    import sys
    script_path = Path(__file__).resolve()  # call resolve() to resolve symlinks (aka shortcuts) if necessary
    sys.path.append(str(script_path.parent.parent))  # two parent directory up

    # Set the dataset name
    data_name = 'my_fashion_mnist'  # 'my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'
    dataset2path = {
        'my_mnist': script_path.parent.parent / 'data/my_mnist',
        'my_fashion_mnist': script_path.parent.parent / 'data/my_fashion_mnist',
        'my_cifar10': script_path.parent.parent / 'data/my_cifar10',
        'my_imagenette': script_path.parent.parent / 'data/my_imagenette',
    }
    print('====================')
    print('dataset_dir =', dataset2path[data_name])

    # Load a huggingface DatasetDict object from disk
    dsd = load_from_disk(str(dataset2path[data_name]))
    ds_train = dsd['train']
    ds_test = dsd['test']
    print(f'{type(ds_train) = }')
    print(f'{type(ds_test) = }')
    print(f'{type(dsd) = } \n ', dsd)


    # %%
    # Print some information of the dataset
    leo = ds_train['label']
    print("ds_train['label']\n  ", type(leo), len(leo), type(leo[0]))

    leo = ds_train['image']
    print("ds_train['image']\n  ", type(leo), len(leo), type(leo[0]))

    leo = ds_train['feature_finetune_vit']
    print("ds_train['feature_finetune_vit']\n  ", type(leo), len(leo), type(leo[0]), len(leo[0]), type(leo[0][0]))

    leo = ds_train['logit_finetune_vit']
    print("ds_train['logit_finetune_vit']\n  ", type(leo), len(leo), type(leo[0]), len(leo[0]), type(leo[0][0]))

    leo = ds_train['feature_pretrain_vit']
    print("ds_train['feature_pretrain_vit']\n  ", type(leo), len(leo), type(leo[0]), len(leo[0]), type(leo[0][0]))


    # %%
    # Resize the images to the same size
    if data_name == 'my_imagenette':
        # One-by-one resize instead of batch tensor resize. Different size tensor can not stack together.
        def resize_transform(examples):
            # images are PIL images (can have different sizes & color like 'imagenette'). See https://huggingface.co/docs/datasets/main/en/image_process#map
            examples["image_resize"] = [image.convert("RGB").resize((320, 320)) for image in examples["image"]]
            return examples
            # my_imagenette in-place resize (examples["image"] = ...examples["image"])
            # Map: 100%|██████████| 9469/9469 [07:17<00:00, 21.29 examples/s]
            # Map: 100%|██████████| 3925/3925 [03:00<00:00, 21.49 examples/s]

            # my_imagenette non in-place resize (examples["image_resize"] = ...examples["image"])
            # Map: 100%|██████████| 9469/9469 [06:55<00:00, 21.42 examples/s]
            # Map: 100%|██████████| 3925/3925 [02:39<00:00, 22.94 examples/s]

        ds_train = ds_train.map(resize_transform, remove_columns=["image"], batched=True)
        ds_test = ds_test.map(resize_transform, remove_columns=["image"], batched=True)
        ds_train = ds_train.rename_column("image_resize", "image")
        ds_test = ds_test.rename_column("image_resize", "image")


    # %%
    # Convert the dataset to torch.Tensor
    ds_train.set_format("torch")
    ds_test.set_format("torch")

    # %%
    leo = ds_train['label']
    print("ds_train['label']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_train['image']
    print("ds_train['image']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_train['feature_finetune_vit']
    print("ds_train['feature_finetune_vit']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_train['logit_finetune_vit']
    print("ds_train['logit_finetune_vit']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_train['feature_pretrain_vit']
    print("ds_train['feature_pretrain_vit']\n  ", type(leo), leo.device, leo.dtype, leo.shape)


    leo = ds_test['label']
    print("ds_test['label']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_test['image']
    print("ds_test['image']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_test['feature_finetune_vit']
    print("ds_test['feature_finetune_vit']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_test['logit_finetune_vit']
    print("ds_test['logit_finetune_vit']\n  ", type(leo), leo.device, leo.dtype, leo.shape)

    leo = ds_test['feature_pretrain_vit']
    print("ds_test['feature_pretrain_vit']\n  ", type(leo), leo.device, leo.dtype, leo.shape)


    # %% Save the dataset to a torch.Tensor file
    dict_tensor = {'train': {'image': ds_train['image'],
                            'label': ds_train['label'],
                            'feature_finetune_vit': ds_train['feature_finetune_vit'],
                            'logit_finetune_vit': ds_train['logit_finetune_vit'],
                            'feature_pretrain_vit': ds_train['feature_pretrain_vit'], },
                'test': {'image': ds_test['image'],
                            'label': ds_test['label'],
                            'feature_finetune_vit': ds_test['feature_finetune_vit'],
                            'logit_finetune_vit': ds_test['logit_finetune_vit'],
                            'feature_pretrain_vit': ds_test['feature_pretrain_vit'], },
                }
    file_path = script_path.parent.parent / f'data/{data_name}_dict_tensor.pt'
    torch.save(dict_tensor, str(file_path))
    print(f'Save the dataset to {file_path}')


exit()

































# %%
def resize_transform(examples):  # images are PIL images (can have different sizes like 'imagenette'). See https://huggingface.co/docs/datasets/main/en/image_process#map
    examples["image_resize"] = [image.convert("RGB").resize((320, 320)) for image in examples["image"]]
    # examples["image_resize"] = [image.resize((320,320)) for image in examples["image"]]
    return examples


# %%

batch_size = 128
num_cpu_worker = 0

dataset_dir = str(download_and_extract(data_name))

dsd = load_from_disk(dataset_dir)  # Load a huggingface DatasetDict object from disk
ds_train = dsd['train'].select_columns(['image', 'feature_finetune_vit', 'label'])  # A huggingface Dataset object. Column: 'image', 'label', 'feature_finetune_vit', 'feature_pretrain_vit', 'logit_finetune_vit'
ds_test = dsd['test'].select_columns(['image', 'feature_finetune_vit', 'label'])


def resize_transform(examples):  # images are PIL images (can have different sizes like 'imagenette'). See https://huggingface.co/docs/datasets/main/en/image_process#map
    examples["image_resize"] = [image.convert("RGB").resize((320, 320)) for image in examples["image"]]
    # examples["image_resize"] = [image.resize((320,320)) for image in examples["image"]]
    return examples


ds_train = ds_train.map(resize_transform, remove_columns=["image"], batched=True)
ds_test = ds_test.map(resize_transform, remove_columns=["image"], batched=True)
ds_train[5]['image_resize']

# %%


# my_mnist
# Map: 100%|██████████| 60000/60000 [01:07<00:00, 883.31 examples/s]
# Map: 100%|██████████| 10000/10000 [00:10<00:00, 911.27 examples/s]
# Map: 100%|██████████| 60000/60000 [00:33<00:00, 1801.85 examples/s]
# Map: 100%|██████████| 10000/10000 [00:05<00:00, 1895.39 examples/s]


# my_fashion_mnist
# Map: 100%|██████████| 60000/60000 [02:04<00:00, 480.92 examples/s]
# Map: 100%|██████████| 10000/10000 [00:20<00:00, 497.82 examples/s]

# my_cifar10
# Map: 100%|██████████| 50000/50000 [04:22<00:00, 190.18 examples/s]
# Map: 100%|██████████| 10000/10000 [00:50<00:00, 197.74 examples/s]


# %%
ds_train[10]['pixel_values']

# %%
image_transform = v2.Compose([
    v2.Resize(size=(30, 30)),  # it can have arbitrary number of leading batch dimensions
    v2.Grayscale(1),
    v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
])
# %%
t0 = time.time()
ds_train.set_format("torch")
temp1 = ds_train['image_resize']
temp2 = ds_train['feature_finetune_vit']
temp3 = ds_train['label']
print(f'Image transform in {time.time() - t0:.2f} seconds')
# %%
type(temp1), type(temp2), type(temp3)
# %%
print(f'{temp1.device}, {temp1.dtype}, {temp1.shape}')
print(f'{temp2.device}, {temp2.dtype}, {temp2.shape}')
print(f'{temp3.device}, {temp3.dtype}, {temp3.shape}')
# %%%

# leo
aaa = torch.randn(100, 3, 4, 5)
bbb = torch.randn(100, 7)
ccc = torch.randn(100)

abc = torch.utils.data.TensorDataset(aaa, bbb, ccc)
ABC = torch.utils.data.StackDataset(aaa=aaa, bbb=bbb, ccc=ccc)

dataset_dict = {'aaa': aaa, 'bbb': bbb, 'ccc': ccc}

my_imagenette_dict_tensor = {'test': {'feature_finetune_vit': 2,
                                      'feature_pretrain_vit': 3,
                                      'image': 1,
                                      'label': 5,
                                      'logit_finetune_vit': 4},
                             'train': {'feature_finetune_vit': 2,
                                       'feature_pretrain_vit': 3,
                                       'image': 1,
                                       'label': 5,
                                       'logit_finetune_vit': 4}
                             }
pprint(my_imagenette_dict_tensor)

# %%
torch.save(dataset_dict, 'my_imagenette_dict_tensor.pt')
leo = torch.load('my_imagenette_dataset_dict.pt', weights_only=True, map_location='cuda:1')

print(f'{type(leo)=}')
for k, v in leo.items():
    print(f'{k=}, {v.device=}, {v.dtype=}, {v.shape=}')

# %%
# ABC[:]['ccc'] = ABC[:]['ccc'].to('cuda:1')


leo = {'rrr': 1, 'ttt': 2, 'yyy': 3}
leo.keys()

# leo = ABC[:]['ccc']
# print(type(leo))
# print(f'{leo.device}, {leo.dtype}, {leo.shape}')

# %%
abc.to('cuda:1')
ABC[0]['aaa'].device, ABC[0]['aaa'].dtype, ABC[0]['aaa'].shape


# %%
leo = torch.stack(temp1)
leo.device, leo.dtype, leo.shape
# %%
i += 1
temp1[i].device, temp1[i].dtype, temp1[i].shape

# %%
torch.utils.data.TensorDataset(temp1, temp2, temp3)

# %%
print(f'{temp1.device}, {temp1.dtype}, {temp1.shape}')
print(f'{temp2.device}, {temp2.dtype}, {temp2.shape}')
print(f'{temp3.device}, {temp3.dtype}, {temp3.shape}')

# %%
temp2 = torch.tensor(ds_train['feature_finetune_vit'])  # Force to break lazy loading
print(f'{temp2.device}, {temp2.dtype}, {temp2.shape}')
temp3 = torch.tensor(ds_train['label'])
print(f'{temp3.device}, {temp3.dtype}, {temp3.shape}')
# %%
ds_train = torch.utils.data.TensorDataset(temp1, temp2, temp3)
print(f'Image transform in {time.time() - t0:.2f} seconds')
# 10.60 sec, 4099 MB


# %%
time2 = time.time()


ds_test = torch.utils.data.TensorDataset(ds_test['image'], ds_test['feature_finetune_vit'], ds_test['label'])
ds_train = torch.utils.data.TensorDataset(ds_train['image'], ds_train['feature_finetune_vit'], ds_train['label'])
time3 = time.time()
print(f'torch.utils.data.TensorDataset in {time3 - time2:.2f} seconds')

# %%
i = -1
# %%
i += 1
bb = aa[i]
bb.device, bb.dtype, bb.shape
# %%


# %%
ds_train.set_format("torch", device='cuda:1')
ds_test.set_format("torch", device='cuda:1')
ds_train = torch.utils.data.TensorDataset(ds_train['image'], ds_train['label'])
ds_test = torch.utils.data.TensorDataset(ds_test['image'], ds_test['label'])
time3 = time.time()
# print(f'set_format in {time3 - time2:.2f} seconds')
print(f'torch.utils.data.TensorDataset in {time3 - time2:.2f} seconds')


train_dataloader = DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    pin_memory_device='cuda:1',
)

test_dataloader = DataLoader(
    ds_test,
    batch_size=batch_size,
    shuffle=False,  # don't need to shuffle test data
    num_workers=0,
    pin_memory=True,
    pin_memory_device='cuda:1',
)

data_info = {
    'data_name': data_name,
    'batch_size': batch_size,
    'train_samples': len(ds_train),
    'test_samples': len(ds_test),
    'train_batches': len(train_dataloader),
    'test_batches': len(test_dataloader),
}

time4 = time.time()
print(f'Create DataLoader in {time4 - time3:.2f} seconds')
