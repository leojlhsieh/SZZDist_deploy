import torch
from torchvision.transforms import v2
from pathlib import Path
import logging


# Create a custom dataset from a dictionary of tensors
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image, label, feature_finetune_vit, transform=None, device=None):
        self.image = image.to(device)
        self.label = label.to(device)
        self.feature_finetune_vit = feature_finetune_vit.to(device)
        self.transform = transform.to(device) if transform else None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.image[idx, :, :, :]  # batch, channel, height, width (10000, 1, 28, 28)
        label = self.label[idx]  # batch (10000)
        feature_finetune_vit = self.feature_finetune_vit[idx, :]  # batch, feature (10000, 768)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label, 'feature_finetune_vit': feature_finetune_vit}


def build_dataloader(data_name: str, bpm_color: str, bpm_width: int, batch_size: int, device=None, small_toy: int = 0):
    # Set the dataset name
    assert data_name in ['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'], f'{data_name = }, only support my_mnist, my_fashion_mnist, my_cifar10, my_imagenette'
    script_path = Path(__file__).resolve()  # call resolve() to resolve symlinks (aka shortcuts) if necessary
    dataset2path = {
        'my_mnist': script_path.parent.parent / 'data/my_mnist_dict_tensor.pt',
        'my_fashion_mnist': script_path.parent.parent / 'data/my_fashion_mnist_dict_tensor.pt',
        'my_cifar10': script_path.parent.parent / 'data/my_cifar10_dict_tensor.pt',
        'my_imagenette': script_path.parent.parent / 'data/my_imagenette_dict_tensor.pt',
    }
    logging.debug('====================')
    logging.debug(f'dataset_dir = {(dataset2path[data_name])}')

    # Load a dictionary of tensors from a file
    dict_tensor = torch.load(dataset2path[data_name], weights_only=True)
    logging.debug(f'{dict_tensor.keys() = }')

    # Print the shape, dtype, and device of the tensors
    dict_tensor_train = dict_tensor['train']
    dict_tensor_test = dict_tensor['test']
    logging.debug('-------------------')
    for k, v in dict_tensor_train.items():
        logging.debug(f'dict_tensor_train {k = }\n  {v.shape = },\n  {v.dtype = },\n  {v.device = }')
    logging.debug('-------------------')
    for k, v in dict_tensor_test.items():
        logging.debug(f'dict_tensor_test {k = }\n  {v.shape = },\n  {v.dtype = },\n  {v.device = }')
    logging.debug('-------------------')

    # Build_transforms
    if bpm_color == 'gray':
        image_transform = v2.Compose([
            v2.Resize(size=(bpm_width, bpm_width)),  # it can have arbitrary number of leading batch dimensions
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            v2.Grayscale(1),
        ])

    elif bpm_color == 'rgb':
        image_transform = v2.Compose([
            # v2.PILToTensor(),
            v2.Resize(size=(bpm_width, bpm_width)),  # it can have arbitrary number of leading batch dimensions
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
        ])

    # Build the dataset
    ds_train = CustomDataset(image=dict_tensor_train['image'],
                             label=dict_tensor_train['label'],
                             feature_finetune_vit=dict_tensor_train['feature_finetune_vit'],
                             transform=image_transform,
                             device=device
                             )
    ds_test = CustomDataset(image=dict_tensor_test['image'],
                            label=dict_tensor_test['label'],
                            feature_finetune_vit=dict_tensor_test['feature_finetune_vit'],
                            transform=image_transform,
                            device=device
                            )
    # Small toy dataset
    if small_toy > 0:
        ds_train = torch.utils.data.Subset(ds_train, range(batch_size*small_toy))
        ds_test = torch.utils.data.Subset(ds_test, range(batch_size*small_toy))

    # Create a dataloader
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Data already on GPU, no need to use CPU multi-process data loading
        pin_memory=False,  # Data already on GPU, no need to use pin_memory
    )
    test_dataloader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=0,
        pin_memory=False,  # RuntimeError: cannot pin 'torch.cuda.ByteTensor' only dense CPU tensors can be pinned
    )

    data_info = {
        'data_name': data_name,
        'batch_size': batch_size,
        'train_samples': len(ds_train),
        'test_samples': len(ds_test),
        'train_batches': len(train_dataloader),
        'test_batches': len(test_dataloader),
    }

    logging.debug(f'{data_info = }')
    return train_dataloader, test_dataloader, data_info


if __name__ == '__main__':
    import time
    import sys
    script_path = Path(__file__).resolve()  # call resolve() to resolve symlinks (aka shortcuts) if necessary
    sys.path.append(str(script_path.parent.parent))  # two parent directory up

    logging.basicConfig(level=logging.DEBUG)

    data_name = 'my_imagenette'  # 'my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'
    bpm_color = 'gray'  # 'gray', 'rgb'
    bpm_width = 300
    batch_size = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    small_toy = 7

    train_dataloader, test_dataloader, data_info = build_dataloader(
        data_name=data_name,
        bpm_color=bpm_color,
        bpm_width=bpm_width,
        batch_size=batch_size,
        device=device,
        small_toy=small_toy
    )

    logging.info(f'{data_info = }')

    for batch, data in enumerate(train_dataloader):
        image = data['image']
        label = data['label']
        feature_finetune_vit = data['feature_finetune_vit']
        logging.debug(f'{batch = }, {image.shape = }')
        logging.debug(f'{batch = }, {image.device = }')
        time.sleep(1)
        if batch == 2:
            break
