# %%
import torch
from torchvision.transforms import v2
from pathlib import Path
import logging


def build_dataloader(data_name: str, batch_size: int = 2, device=None, small_toy: int = 0):
    # Set the dataset name
    script_path = Path(__file__).resolve()  # call resolve() to resolve symlinks (aka shortcuts) if necessary
    dataset2path = {
        'my_mnist': script_path.parent.parent / 'data/my_mnist_dict_tensor.pt',
        'my_fashion_mnist': script_path.parent.parent / 'data/my_fashion_mnist_dict_tensor.pt',
        'my_cifar10': script_path.parent.parent / 'data/my_cifar10_dict_tensor.pt',
        'my_imagenette': script_path.parent.parent / 'data/my_imagenette_dict_tensor.pt',
    }
    logging.debug('====================')
    logging.debug('dataset_dir =', dataset2path[data_name])

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

    # Create a dataset on device.  https://pytorch.org/docs/stable/data.html#torch.utils.data.StackDataset
    selected_dict_tensor_train = {
        'image': dict_tensor_train['image'].detach().clone().to(device),
        'label': dict_tensor_train['label'].detach().clone().to(device),
        'feature_finetune_vit': dict_tensor_train['feature_finetune_vit'].detach().clone().to(device),
    }
    selected_dict_tensor_test = {
        'image': dict_tensor_test['image'].detach().clone().to(device),
        'label': dict_tensor_test['label'].detach().clone().to(device),
        'feature_finetune_vit': dict_tensor_test['feature_finetune_vit'].detach().clone().to(device),
    }





    ds_train = torch.utils.data.StackDataset(**selected_dict_tensor_train)
    ds_test = torch.utils.data.StackDataset(**selected_dict_tensor_test)

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


    logging.basicConfig(level=logging.INFO)

    data_name = 'my_cifar10'  # 'my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'
    batch_size = 128
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    small_toy = 7
    train_dataloader, test_dataloader, data_info = build_dataloader(data_name, batch_size, device, small_toy)

    logging.info(f'{data_info = }')

    image_transform = v2.Compose([
            # v2.PILToTensor(),
            v2.Grayscale(1),
            v2.Resize(size=(300, 300)),  # it can have arbitrary number of leading batch dimensions
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            # v2.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
        ])



    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),  # skip 1 batch, warmup 2 batch, profile 3 batches. Need to run at least (1+2+3)*1=6 batches in total.
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
    # ===== Training batch loop =====
        logging.info('===== Training batch loop =====')
        for batch, data in enumerate(train_dataloader):
            t0 = time.time()
            # logging.info(f"{data['image'].device = }, {data['feature_finetune_vit'].device = }, {data['label'].device = }")
            prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
            with torch.profiler.record_function("image to CUDA & transform"):
                image = image_transform(data['image'])
            t1 = time.time()
            logging.info(f"Seconds: {t1-t0:.2f}")
            with torch.profiler.record_function("feature & label to CUDA"):
                feature = data['feature_finetune_vit']
                label = data['label']


    exit()