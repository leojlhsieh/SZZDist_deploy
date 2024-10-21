# %%
# import torch.optim as optim
# import torch.nn.functional as F


# import torch.nn as nn
# from pathlib import Path
# from tqdm.auto import tqdm
# from tool.download_data import download_and_extract
# from datasets import load_from_disk  # Huggingface datasets  # pip install datasets
# from torch.utils.data import Dataset  # For custom dataset of PyTorch
# from torch.utils.data import DataLoader  # Easy to get datasets in batches, shuffle, multiprocess, etc.
# from torch.profiler import profile, record_function, ProfilerActivity

# ==== Keep import ====
import torch
from torchvision.transforms import v2
import time
import pathlib

import random
import numpy as np

from tool.check_gpu import check_gpu
from tool.build_dataloader import build_dataloader
from model.leo_model_v20241015 import build_model

from pprint import pprint, pformat
import os
from datetime import datetime
import wandb
from torcheval.metrics.functional import multiclass_accuracy


from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s %(message)s - Line: %(lineno)d',
    datefmt="%H:%M:%S",
    )

logging.info("================================")
logging.info("================================")
logging.info(f'Running {__file__}')
logging.info("Start logging...")
# Get the arguments
parser = ArgumentParser()
parser.add_argument("--temp1", type=str, default='ccc')
parser.add_argument("--machine_name", type=str, default='musta_3090Ti', choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])  # , required=True)
parser.add_argument("--entity_name", type=str, default='leohsieh-epfl')
parser.add_argument("--project_name", type=str, default='szzbpm-distil-4')
parser.add_argument("--sweep_id", type=str, default='8li9kx8l', help="If not provided, use a empty sweep created by leo")

parser.add_argument("--data_name", type=str, default='my_cifar10', choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])  # , required=True)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--loss_fn_1", type=str, default='HuberLoss')
parser.add_argument("--loss_fn_2", type=str, default='CrossEntropyLoss')
parser.add_argument("--loss_ratio", type=float, default=0.8, help="loss_total = loss_ratio * loss_fn_1 + (1 - loss_ratio) * loss_fn_2")
parser.add_argument("--lr_bpm", type=float, default=1e-03)
parser.add_argument("--lr_feature", type=float, default=1e-03)
parser.add_argument("--lr_class", type=float, default=1e-03)
parser.add_argument("--lr_scheduler", type=str, default='OneCycleLR')
parser.add_argument("--optimizer", type=str, default='adamW')
parser.add_argument("--bpm_color", type=str, default='gray', choices=['gray', 'rgb'])
parser.add_argument("--bpm_mode", type=str, default='bpm', choices=['bpm', 'CNNpatch-bpm', 'fft-bpm', 'nothing'])
parser.add_argument("--bpm_depth", type=int, default=4, choices=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument("--bpm_width", type=int, default=300, choices=[75, 150, 300, 600, 1200])
parser.add_argument("--bpm_parallel", type=int, default=1)
parser.add_argument("--model_feature", type=str, default='maxpool25-ReLU', choices=['maxpool25-ReLU', 'CNN-ReLU', 'rearange', 'nothing'])

parser.add_argument("--small_toy", type=int, default=0, help="Use a small dataset of `small_toy` batches for debugging. 0 means full dataset")
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--save_total_limit", type=int, default=4)
parser.add_argument("--num_cpu_worker", type=int, default=0)
parser.add_argument("--f", type=str, help="If run in Jupyter Notebook, add this line to avoid error")
args = parser.parse_args()
logging.debug(f'vars(args) = {pformat(vars(args), sort_dicts=False)}')






# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


# Device configuration
if args.machine_name == 'musta_2080Ti':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

check_gpu()
logging.info(f'{device = }')
logging.info(f"{os.cpu_count() = }, {args.num_cpu_worker = }")



time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wanb_runname = f'{time_stamp}--{args.bpm_width}x{args.bpm_depth}--{args.data_name}--{args.machine_name}'
vars(args).update({'wanb_runname': wanb_runname})
vars(args).update({'__file__': __file__})
logging.info(f'{args.wanb_runname = }')  # Example: "musta_3090Ti--2024-10-17_15-09-58"
logging.info(f'{args.__file__ = }')  # Example: "musta_3090Ti--2024-10-17_15-09-58"

# all args are set, check what they are
logging.debug(f'vars(args) = {pformat(vars(args), sort_dicts=False)}')







# ===== Some helper functions =====
def build_transforms(bpm_color, bpm_width):
    if bpm_color == 'gray':
        image_transform = v2.Compose([
            # v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            v2.Grayscale(1),
            v2.Resize(size=(bpm_width, bpm_width)),  # it can have arbitrary number of leading batch dimensions
        ])
    elif bpm_color == 'rgb':
        image_transform = v2.Compose([
            # v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),  # scale=True: 0-255 => 0-1
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor image from [0, 1] to [-1, 1]. image = (image - mean) / std.
            # v2.Grayscale(1),
            v2.Resize(size=(bpm_width, bpm_width)),  # it can have arbitrary number of leading batch dimensions
        ])

    return image_transform





# %%  Step 3: Define your machine learning code

# %%  Step 2️: Start wandb
os.environ["WANDB_API_KEY"] = 'c08ff557ed402c26a0c57ca0e7803529bdba9268'
wandb.login()
full_sweep_id = f'{args.entity_name}/{args.project_name}/{args.sweep_id}'


def train_by_wandb():
    with wandb.init(config=vars(args), name=args.wanb_runname):
        config = wandb.config  # sweep agent will overwrite if already exists in args
        logging.info(f'wandb config = {pformat(dict(config), sort_dicts=False)}')

        # ==== Build model, loss, optimizer, scheduler, dataset, dataloader ====
        # choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette']
        image_transform = build_transforms(config.bpm_color, config.bpm_width).to(device)
        train_loader, test_loader, data_info = build_dataloader(config.data_name, config.batch_size, device=device, small_toy=config.small_toy)
        model_bpm, model_feature, model_classifier = build_model(config.bpm_color, config.bpm_mode, config.bpm_depth, config.bpm_width, config.bpm_parallel, config.model_feature, device=device)
        logging.info(f'data_info = {pformat(data_info, sort_dicts=False)}')

        loss_fn_1 = torch.nn.HuberLoss().to(device)
        loss_fn_2 = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW([
            {'params': model_bpm.parameters(), 'lr': config.lr_bpm},
            {'params': model_feature.parameters(), 'lr': config.lr_feature},
            {'params': model_classifier.parameters(), 'lr': config.lr_class},
        ], weight_decay=0.01)  # 'AdamW' object has no attribute 'to'
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.lr_bpm, config.lr_feature, config.lr_class],
            epochs=config.epochs,
            steps_per_epoch=data_info['train_batches'],
            pct_start=0.1,
            anneal_strategy='linear',
            div_factor=8,
            final_div_factor=256)  # 'OneCycleLR' object has no attribute 'to'
        loss_ratio = torch.tensor(config.loss_ratio).to(device)
        one = torch.tensor(1.0).to(device)

        # ==== Epoch loop ====
        saved_checkpoint = [{'acc': 0.00, 'path': None},]
        for epoch in range(config.epochs):
            time_start = time.time()
            logging.info(f"---------- Epoch: {epoch}----------")
            # ===== Training =====
            model_bpm.train()
            model_feature.train()
            model_classifier.train()
            train_loss_1_epoch = 0
            train_loss_2_epoch = 0
            train_loss_total_epoch = 0
            # ===== Training batch loop =====
            for batch, data in enumerate(train_loader):
                image = image_transform(data['image'])
                feature = data['feature_finetune_vit']
                label = data['label']

                # Forward pass
                camera_pred = model_bpm(image)
                feature_pred = model_feature(camera_pred)
                logit_pred = model_classifier(feature_pred)

                loss_1 = loss_fn_1(feature_pred, feature)
                loss_2 = loss_fn_2(logit_pred, label)  # For Pytorch CrossEntropyLoss, the input is expected to contain raw. Do NOT apply softmax and argmax before passing it to the loss function.
                loss_total = loss_ratio * loss_1 + (one - loss_ratio) * loss_2


                loss_1_item = loss_1.item()
                loss_2_item = loss_2.item()
                loss_total_item = loss_total.item()  # Reduce GUP to CPU transfer

                train_loss_1_epoch += loss_1_item
                train_loss_2_epoch += loss_2_item
                train_loss_total_epoch += loss_total_item

                # Backward pass
                optimizer.zero_grad()
                loss_total.backward()

                # Step with optimizer
                optimizer.step()
                scheduler.step()  # For OneCycleLR, step() should be invoked after each batch instead of after each epoch

                if batch % 75 == 0:
                    wandb.log({"every_75_train_batches/loss_1": loss_1_item})
                    wandb.log({"every_75_train_batches/loss_2": loss_2_item})
                    wandb.log({"every_75_train_batches/loss_total": loss_total_item})

            train_loss_1_epoch /= len(train_loader)
            train_loss_2_epoch /= len(train_loader)
            train_loss_total_epoch /= len(train_loader)
            logging.info(f"TRAIN loss 1: {train_loss_1_epoch:.5f}, loss 2: {train_loss_2_epoch:.5f}, loss total: {train_loss_total_epoch:.5f}")
            wandb.log({"epoch": epoch, "train/loss_1": train_loss_1_epoch})
            wandb.log({"epoch": epoch, "train/loss_2": train_loss_2_epoch})
            wandb.log({"epoch": epoch, "train/loss_total": train_loss_total_epoch})
            for name, data in model_bpm.named_parameters():
                wandb.log({"epoch": epoch, f"parameters/model_bpm.{name}": wandb.Histogram(data.detach().cpu().numpy())})
            for name, data in model_feature.named_parameters():
                wandb.log({"epoch": epoch, f"parameters/model_feature.{name}": wandb.Histogram(data.detach().cpu().numpy())})
            for name, data in model_classifier.named_parameters():
                wandb.log({"epoch": epoch, f"parameters/model_classifier.{name}": wandb.Histogram(data.detach().cpu().numpy())})

            # ===== Testing =====
            model_bpm.eval()  # important for dropout and batch normalization layers
            model_feature.eval()
            model_classifier.eval()
            test_loss_1_epoch = 0
            test_loss_2_epoch = 0
            test_loss_total_epoch = 0
            test_accuracy_epoch = 0
            # Turn on inference context manager
            with torch.inference_mode():
                # ===== Testing batch loop =====
                for batch, data in enumerate(test_loader):
                    image = image_transform(data['image'])
                    feature = data['feature_finetune_vit']
                    label = data['label']

                    # Forward pass
                    camera_pred = model_bpm(image)
                    feature_pred = model_feature(camera_pred)
                    logit_pred = model_classifier(feature_pred)
                    label_pred = logit_pred.argmax(dim=1)

                    loss_1 = loss_fn_1(feature_pred, feature)
                    loss_2 = loss_fn_2(logit_pred, label)  # For Pytorch CrossEntropyLoss, the input is expected to contain raw. Do NOT apply softmax and argmax before passing it to the loss function.
                    loss_total = loss_ratio * loss_1 + (one - loss_ratio) * loss_2

                    test_accuracy = multiclass_accuracy(label_pred, target=label)

                    test_loss_1_epoch += loss_1.item()
                    test_loss_2_epoch += loss_2.item()
                    test_loss_total_epoch += loss_total.item()
                    test_accuracy_epoch += test_accuracy.item()  # Ony onec, no need to reduce GUP to CPU transfer
            test_loss_1_epoch /= len(test_loader)
            test_loss_2_epoch /= len(test_loader)
            test_loss_total_epoch /= len(test_loader)
            test_accuracy_epoch /= len(test_loader)
            runtime_minute = (time.time() - time_start) / 60.0
            logging.info(f"TEST loss 1: {test_loss_1_epoch:.5f}, loss 2: {test_loss_2_epoch:.5f}, loss total: {test_loss_total_epoch:.5f}")
            logging.info(f"TEST accuracy: {test_accuracy_epoch:.5f}")
            logging.info(f"One epoch runt {runtime_minute:.2f} minutes, train {data_info['train_batches']} batches, test {data_info['test_batches']} batches")
            wandb.log({"epoch": epoch, "test/loss_1": test_loss_1_epoch})
            wandb.log({"epoch": epoch, "test/loss_2": test_loss_2_epoch})
            wandb.log({"epoch": epoch, "test/loss_total": test_loss_total_epoch})
            wandb.log({"epoch": epoch, "test/accuracy": test_accuracy_epoch})
            wandb.log({"epoch": epoch, "runtime/minute": runtime_minute})

            # ===== Save model checkpoint =====
            checkpoint_dir = pathlib.Path(__file__).parent / 'checkpoint' / f'{config.wanb_runname}'
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            checkpoint_file_dir = checkpoint_dir / f'acc{test_accuracy_epoch:.5f}-epoch{epoch:03d}-checkpoint.pth'
            # If accuracy >= the max value of all current 'acc' in saved_checkpoint, append
            if test_accuracy_epoch >= max([x['acc'] for x in saved_checkpoint]):
                # Save model state dict
                total_model_state_dict = {
                    'model_bpm.state_dict': model_bpm.state_dict(),
                    'model_feature.state_dict': model_feature.state_dict(),
                    'model_classifier.state_dict': model_classifier.state_dict(),
                    'wandb_config': dict(config),  # Save the config used in this run
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(total_model_state_dict, f=str(checkpoint_file_dir))
                saved_checkpoint.append({'acc': test_accuracy_epoch, 'path': checkpoint_file_dir})
                # If the length of saved_checkpoint > save_total_limit, remove the one with the smallest 'acc'
                if len(saved_checkpoint) > args.save_total_limit:
                    to_delete = min(saved_checkpoint, key=lambda x: x['acc'])
                    if to_delete['path'] is not None:
                        to_delete['path'].unlink()  # Delete the file
                    saved_checkpoint.remove(to_delete)
            logging.debug(f"Saved checkpoint: {checkpoint_file_dir}")
        logging.info("All epochs completed!")
    logging.debug("Exit wandb.init()")


#   Step 4: Activate sweep agents
wandb.agent(full_sweep_id, train_by_wandb, count=1)
wandb.finish()
logging.info("wandb.finish() completed!")

# %%
