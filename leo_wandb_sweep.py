# %%
from pprint import pprint
import wandb
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from tqdm.auto import tqdm
from datetime import datetime
from tool.check_gpu import check_gpu
from datasets import load_from_disk  # Huggingface datasets  # pip install datasets
from torch.utils.data import Dataset  # For custom dataset of PyTorch
from torch.utils.data import DataLoader  # Easy to get datasets in batches, shuffle, multiprocess, etc.
from argparse import ArgumentParser
from model.leo_model_v20241015 import build_model
from torcheval.metrics.functional import multiclass_accuracy

# Get the arguments
parser = ArgumentParser()
parser.add_argument("--machine_name", type=str, choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_H100', 'kuma_H100'], default='musta_2080Ti', required=True)
parser.add_argument("--data_name", type=str, choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'], default='my_cifar10', required=True)
parser.add_argument("--sweep_id", type=str)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument("--small_toy", action="store_true", default=True, help="Use a small toy dataset for debugging")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()
print(f'{args = }')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


# Device configuration
if args.machine_name == 'musta_2080Ti':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
check_gpu()
print(f'{device = }')

machine_name = args.machine_name
time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wanb_name = f'{machine_name}--{time_stamp}'
print(f'{wanb_name = }')

NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0
print(f"{NUM_WORKERS = }")


# %%  Step 1️: Define a sweep configuration
entity_name = 'leohsieh-epfl'
project_name = 'szzbpm-distil-3'

if args.sweep_id is None:
    sweep_config = {
        'entity': entity_name,
        'project': project_name,
        'method': 'random',
        'metric': {'goal': 'maximize', 'name': 'test/accuracy'},
        'parameters': {'data_name': {'value': args.data_name},  # choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette']
                       'epochs': {'value': args.epochs},
                       'batch_size': {'value': args.batch_size},
                       'loss_fn_1': {'value': 'HuberLoss'},
                       'loss_fn_2': {'value': 'CrossEntropyLoss'},
                       'loss_ratio': {'distribution': 'uniform',  # loss_total = loss_ratio * loss_fn_1 + (1 - loss_ratio) * loss_fn_2
                                      'max': 1,
                                      'min': 0},
                       'lr_bpm': {'distribution': 'log_uniform_values',
                                  'max': 1e-02,
                                  'min': 1e-05},
                       'lr_feature': {'distribution': 'log_uniform_values',
                                      'max': 1e-02,
                                      'min': 1e-05},
                       'lr_class': {'distribution': 'log_uniform_values',
                                    'max': 1e-02,
                                    'min': 1e-05},
                       'lr_scheduler': {'value': 'OneCycleLR'},
                       'optimizer': {'value': 'adamW'},
                       'bpm_color': {'value': 'gray'},  # 'gray', 'rgb'
                       'bpm_mode': {'value': 'bpm'},  # 'bpm', 'CNNpatch-bpm', 'fft-bpm', 'nothing'
                       'bpm_depth': {'value': 4},  # 1, 2, 3, 4, 5, 6, 7, 8
                       'bpm_width': {'value': 300},  # 75, 150, 300, 600, 1200
                       'bpm_parallel': {'value': 1},  # 1, 3
                       'model_feature': {'value': 'maxpool30-ReLU'},  # 'CNN-ReLU', 'rearange', 'nothing'
                       }
    }
    pprint(sweep_config)


# %%  Step 2️: Initialize the Sweep
wandb.login(key='c08ff557ed402c26a0c57ca0e7803529bdba9268')

if args.sweep_id is None:
    sweep_id = wandb.sweep(sweep_config)
else:
    sweep_id = args.sweep_id
sweep_id = f'{entity_name}/{project_name}/{sweep_id}'
print(f'{sweep_id = }')


# %%  Step 3: Define your machine learning code
def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, name=wanb_name):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        image_transform = build_transforms(config.bpm_color, device=device)
        tarin_loader, test_loader, data_info = build_dataset(config.data_name, config.batch_size)
        model_bpm, model_feature, model_class = build_model(config.bpm_color, config.bpm_mode, config.bpm_depth, config.bpm_width, config.bpm_parallel, config.model_feature, device=device)
        pprint(data_info)

        loss_fn_1 = nn.HuberLoss().to(device)
        loss_fn_2 = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW([
            {'params': model_bpm.parameters(), 'lr': config.lr_bpm},
            {'params': model_feature.parameters(), 'lr': config.lr_feature},
            {'params': model_class.parameters(), 'lr': config.lr_class},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.lr_bpm, config.lr_feature, config.lr_class],
            epochs=config.epochs,
            steps_per_epoch=data_info['train_batches'],
            pct_start=0.1,
            anneal_strategy='linear',
            div_factor=8,
            final_div_factor=256)

        for epoch in range(config.epochs):
            print(f"---------- Epoch: {epoch}----------")
            # Training
            wandb.watch(model_bpm)
            model_bpm.train()
            model_feature.train()
            model_class.train()
            train_loss_1_epoch = 0
            train_loss_2_epoch = 0
            train_loss_total_epoch = 0
            # Add a loop to loop through training batches
            for batch, data in enumerate(tarin_loader):
                image = image_transform(data['image'].to(device))
                feature = data['feature_finetune_vit'].to(device)
                label = data['label'].to(device)

                # Forward pass
                camera_pred = model_bpm(image)
                feature_pred = model_feature(camera_pred)
                logit_pred = model_class(feature_pred)

                loss_1 = loss_fn_1(feature_pred, feature)
                loss_2 = loss_fn_2(logit_pred, label)  # For Pytorch CrossEntropyLoss, the input is expected to contain raw. Do NOT apply softmax and argmax before passing it to the loss function.
                loss_total = config.loss_ratio * loss_1 + (1 - config.loss_ratio) * loss_2

                train_loss_1_epoch += loss_1.item()
                train_loss_2_epoch += loss_2.item()
                train_loss_total_epoch += loss_total.item()
                # Backward pass
                optimizer.zero_grad()
                loss_total.backward()

                # Step with optimizer
                optimizer.step()
                scheduler.step()  # For OneCycleLR, step() should be invoked after each batch instead of after each epoch
            train_loss_1_epoch /= len(tarin_loader)
            train_loss_2_epoch /= len(tarin_loader)
            train_loss_total_epoch /= len(tarin_loader)
            print(f"Train Loss 1: {train_loss_1_epoch:.5f}, Train Loss 2: {train_loss_2_epoch:.5f}, Train Loss Total: {train_loss_total_epoch:.5f}")
            wandb.log({"epoch": epoch, "train/loss_1": train_loss_1_epoch})
            wandb.log({"epoch": epoch, "train/loss_2": train_loss_2_epoch})
            wandb.log({"epoch": epoch, "train/loss_total": train_loss_total_epoch})

            # Testing
            model_bpm.eval()
            model_feature.eval()
            model_class.eval()
            test_loss_1_epoch = 0
            test_loss_2_epoch = 0
            test_loss_total_epoch = 0
            test_accuracy_epoch = 0
            test_f1_epoch = 0
            # Turn on inference context manager
            with torch.inference_mode():
                for batch, data in enumerate(test_loader):
                    image = image_transform(data['image'].to(device))
                    feature = data['feature_finetune_vit'].to(device)
                    label = data['label'].to(device)

                    # Forward pass
                    camera_pred = model_bpm(image)
                    feature_pred = model_feature(camera_pred)
                    logit_pred = model_class(feature_pred)
                    label_pred = logit_pred.argmax(dim=1)

                    loss_1 = loss_fn_1(feature_pred, feature)
                    loss_2 = loss_fn_2(logit_pred, label)  # For Pytorch CrossEntropyLoss, the input is expected to contain raw. Do NOT apply softmax and argmax before passing it to the loss function.
                    loss_total = config.loss_ratio * loss_1 + (1 - config.loss_ratio) * loss_2
                    test_accuracy = multiclass_accuracy(label_pred, target=label)

                    test_loss_1_epoch += loss_1.item()
                    test_loss_2_epoch += loss_2.item()
                    test_loss_total_epoch += loss_total.item()
                    test_accuracy_epoch += test_accuracy.item()
                test_loss_1_epoch /= len(test_loader)
                test_loss_2_epoch /= len(test_loader)
                test_loss_total_epoch /= len(test_loader)
                test_accuracy_epoch /= len(test_loader)
                print(f"Test Loss 1: {test_loss_1_epoch:.5f}, Test Loss 2: {test_loss_2_epoch:.5f}, Test Loss Total: {test_loss_total_epoch:.5f}")
                print(f"Test Accuracy: {test_accuracy_epoch:.5f}")
                wandb.log({"epoch": epoch, "test/loss_1": test_loss_1_epoch})
                wandb.log({"epoch": epoch, "test/loss_2": test_loss_2_epoch})
                wandb.log({"epoch": epoch, "test/loss_total": test_loss_total_epoch})
                wandb.log({"epoch": epoch, "test/accuracy": test_accuracy_epoch})
        print("All epochs completed!")


def build_transforms(bpm_color, device=None):
    if bpm_color == 'gray':
        image_transform = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            v2.Resize(size=(300, 300)),  # it can have arbitrary number of leading batch dimensions
            v2.Grayscale(1),
        ])
    elif bpm_color == 'rgb':
        image_transform = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            v2.Resize(size=(300, 300)),  # it can have arbitrary number of leading batch dimensions
        ])

    return image_transform.to(device)


def build_dataset(data_name, batch_size):
    data2path = {
        'mnist': "./data/mnist",
        'fashion_mnist': "./data/fashion_mnist",
        'cifar10': "./data/cifar10",
        'imagenette': './data/imagenette',

        'my_mnist': './data/my_mnist',
        'my_fashion_mnist': './data/my_fashion_mnist',
        'my_cifar10': './data/my_cifar10',
        'my_imagenette': './data/my_imagenette',
    }

    ds = load_from_disk(data2path[data_name]).with_format("torch")  # or .with_format("torch", device=device)
    ds_train = ds['train'].select_columns(['image', 'feature_finetune_vit', 'label'])  # 'image', 'label', 'feature_finetune_vit', 'feature_pretrain_vit', 'logit_finetune_vit'
    ds_test = ds['test'].select_columns(['image', 'feature_finetune_vit', 'label'])  # 'image', 'label', 'feature_finetune_vit', 'feature_pretrain_vit', 'logit_finetune_vit'
    if args.small_toy:
        ds_train = ds_train.select(range(args.batch_size*3+1))
        ds_test = ds_test.select(range(args.batch_size*3+1))

    train_dataloader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    data_info = {
        'batch_size': batch_size,
        'train_samples': len(ds_train),
        'test_samples': len(ds_test),
        'train_batches': len(train_dataloader),
        'test_batches': len(test_dataloader),
    }

    return train_dataloader, test_dataloader, data_info







# %%  Step 4: Activate sweep agents
wandb.agent(sweep_id, train, count=5)

wandb.finish()




# %%
