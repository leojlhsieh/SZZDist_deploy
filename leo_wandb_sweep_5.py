import torch
from torchvision.transforms import v2
import time
from pathlib import Path

from tqdm.auto import tqdm
import random
import numpy as np
import os
import logging

from tool.check_gpu import check_gpu
from tool.build_dataloader import build_dataloader
from model.leo_model_v20241021 import build_model

from pprint import pprint, pformat
from datetime import datetime
import wandb
from torcheval.metrics.functional import multiclass_accuracy
import json


from argparse import ArgumentParser


# Ensure deterministic behavior (not cross platforms)
# I choose speed because I'll run on different device anyway. # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
torch.backends.cudnn.deterministic = False  # False is faster, True is more reproducible
torch.backends.cudnn.benchmark = True  # True is faster, False is more reproducible
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s %(message)s - Line: %(lineno)d',
                    datefmt="%H:%M:%S",
                    )

script_path = Path(__file__).resolve()  # call resolve() to resolve symlinks (aka shortcuts) if necessary
logging.info("================================")
logging.info("================================")
logging.info(f'Running {str(script_path)}')
logging.info("Start logging...")
# Get the arguments
parser = ArgumentParser()
parser.add_argument("--temp1", type=str, default='ccc')
parser.add_argument("--machine_name", type=str, default='musta_3090Ti', choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])  # , required=True)
parser.add_argument("--entity_name", type=str, default='leohsieh-epfl')
parser.add_argument("--project_name", type=str, default='szzbpm-distil-5')
parser.add_argument("--sweep_id", type=str, default=None, help="If not provided, use a empty sweep created by leo(random-93e3dtbc)")

parser.add_argument("--data_name", type=str, default='my_imagenette', choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])  # , required=True)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--loss_fn_1", type=str, default='HuberLoss')
parser.add_argument("--loss_fn_2", type=str, default='CrossEntropyLoss')
parser.add_argument("--loss_ratio", type=float, default=0.5, help="Total loss = loss_ratio * L1 + (1-loss_ratio) * L2")
parser.add_argument("--lr_bpm", type=float, default=5e-03, help="nPOLO paper use lr=5e-03 for optics")
parser.add_argument("--lr_feature", type=float, default=2.5e-03, help="nPOLO paper use lr=2.5e-04 for digital classifier")
parser.add_argument("--lr_class", type=float, default=2.5e-03, help="nPOLO paper use lr=2.5e-04 for digital classifier")
parser.add_argument("--lr_scheduler", type=str, default='OneCycleLR')
parser.add_argument("--optimizer", type=str, default='adamW')
parser.add_argument("--bpm_color", type=str, default='gray', choices=['gray', 'rgb'])
parser.add_argument("--bpm_mode", type=str, default='bpm', choices=['bpm', 'CNNpatch-bpm', 'fft-bpm', 'nothing'])
parser.add_argument("--bpm_depth", type=int, default=4, choices=[2, 3, 4, 5, 6, 7, 8])
parser.add_argument("--bpm_width", type=int, default=300, choices=[75, 150, 300, 600, 1200])
parser.add_argument("--Ldist", type=float, default=6e-03, choices=[3e-3, 6e-3, 12e-3], help="Distance between SLM & mirror [m]")
parser.add_argument("--bpm_parallel", type=int, default=1)
parser.add_argument("--feature_mode", type=str, default='avgpool25-ReLU', choices=['avgpool25-ReLU', 'CNN-ReLU', 'rearange', 'nothing'])

parser.add_argument("--small_toy", type=int, default=0, help="Use a small dataset of `small_toy` batches for debugging. 0 means full dataset")
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--save_total_limit", type=int, default=4)
parser.add_argument("--num_cpu_worker", type=int, default=0)
parser.add_argument("--f", type=str, help="If run in Jupyter Notebook, add this line to avoid error")
args = parser.parse_args()
logging.debug(f'vars(args) = {pformat(vars(args), sort_dicts=False)}')


# ===== Check the arguments =====
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


time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")  # Use time_stamp down to nanosecond, so it's unique when running multiple tasks on Kuma cluster
wanb_runname = f'{args.machine_name}---{time_stamp}--{args.data_name}'
vars(args).update({'wanb_runname': wanb_runname})
vars(args).update({'script_path': str(script_path)})
logging.info(f'{args.wanb_runname = }')  # Example: "musta_3090Ti--2024-10-17_15-09-58.123456"
logging.info(f'{args.script_path = }')
if args.sweep_id is None:
    sweep_id_dict = {'my_mnist': '6hmykyxr',
                     'my_fashion_mnist': 'js4zojc3',
                     'my_cifar10': 'lwi954r2',
                     'my_imagenette': '6vj8qy15',
                     }
    vars(args).update({'sweep_id': sweep_id_dict[args.data_name]})
full_sweep_id = f'{args.entity_name}/{args.project_name}/{args.sweep_id}'


# all args are set, check what they are
logging.debug(f'vars(args) = {pformat(vars(args), sort_dicts=False)}')


# ===== Some helper functions =====
def get_sweep_dict(sweep_index: int) -> dict:
    shuffled_combinations_json_dir = script_path.parent / 'tool/shuffled_combinations_first_most_demanding.json'
    with open(shuffled_combinations_json_dir, 'r') as json_file:
        combinations = json.load(json_file)
    sweep_dict = combinations[sweep_index]
    return sweep_dict


# ==== Main function to be called by wandb sweep agent ====
def called_by_wandb_sweep_agent():
    with wandb.init(config=vars(args), name=args.wanb_runname):  # sweep agent will overwrite if already exists in args
        config = wandb.config  # Access all hyperparameters through wandb.config, so logging matches execution!
        sweep_index = config.sweep_index  # sweep agent give this sweeping number. value from 0 to len(combinations)-1, in order.
        sweep_dict = get_sweep_dict(sweep_index)
        logging.info(f'sweep_index = {sweep_index}, sweep_dict = {pformat(sweep_dict, sort_dicts=False)}')
        wandb.config.update(sweep_dict, allow_val_change=True)  # Check the note below to see what's in sweep_dict
        wandb.config.update({'lr_feature': sweep_dict['lr_class']}, allow_val_change=True)  # Make sure lr_feature is the same as lr_class
        config = wandb.config  # Access all hyperparameters through wandb.config, so logging matches execution!
        # Note:
        # hyperparameters = {
        #     'bpm_depth': [3, 4, 6],  # Number of SLM modulation (aka bounce) in BPM
        #     'bpm_width': [150, 300, 600],  # Width of SLM [pixel], height is same as width
        #     'Ldist': [3e-3, 6e-3, 12e-3],  # Distance between SLM & mirror [m]
        #     'lr_bpm': log_uniform_distribution(start=5e-3 * 0.1, end=5e-3 * 10, points=7),  # nPOLO paper use lr=5e-3 for optics
        #     'lr_class': log_uniform_distribution(start=2.5e-4 * 0.1, end=2.5e-4 * 10, points=7),  # nPOLO paper use lr=2.5e-4 for digital classifier
        #     'loss_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Total loss = loss_ratio * L1 + (1-loss_ratio) * L2
        # }

        # ==== Build model, loss, optimizer, scheduler, dataset, dataloader ====
        train_loader, test_loader, data_info = build_dataloader(
            data_name=config.data_name,
            bpm_color=config.bpm_color,
            bpm_width=config.bpm_width,
            batch_size=config.batch_size,
            device=device,
            small_toy=config.small_toy
        )
        # model_bpm.to(device) # <<< This will cause error, model_bpm.to(device) is currelely not working. Please assign device to build_model_bpm function's argument
        model_bpm, model_feature, model_classifier, bpm_inof = build_model(
            bpm_color=config.bpm_color,
            bpm_mode=config.bpm_mode,
            bpm_depth=config.bpm_depth,
            bpm_width=config.bpm_width,
            Ldist=config.Ldist,
            bpm_parallel=config.bpm_parallel,
            feature_mode=config.feature_mode,
            device=device
        )
        logging.info(f'data_info = {pformat(data_info, sort_dicts=False)}')
        logging.info(f'bpm_inof = {pformat(bpm_inof, sort_dicts=False)}')
        wandb.config.update(data_info, allow_val_change=True)  # Check the note below to see what's in data_info
        wandb.config.update(bpm_inof, allow_val_change=True)  # Check the note below to see what's in bpm_inof
        # # Note:
        # data_info = {
        #     'data_name': data_name,
        #     'batch_size': batch_size,
        #     'train_samples': len(ds_train),
        #     'test_samples': len(ds_test),
        #     'train_batches': len(train_dataloader),
        #     'test_batches': len(test_dataloader),
        # }
        # bpm_info = {
        #     'bpm_Nx_dx_Lx': (Nx_bpm, dx_bpm, Lx_bpm),
        #     'bpm_Lz_dz_Nz': (Lz_bpm, dz_bpm, Nz_bpm),
        #     'slm_Nx_dx_Lx': (Nx_slm, dx_slm, Lx_slm),
        #     'cam_Nx_dx_Lx': (Nx_cam, dx_cam, Lx_cam),
        #     'layer_sampling': layer_sampling,
        #     'Ldist': Ldist,
        #     'bpm_depth': bpm_depth,
        #     'bpm_width': bpm_width,
        # }

        # ==== Count the number of trainable parameters ====
        bpm_params_count = sum(p.numel() for p in model_bpm.parameters() if p.requires_grad)  # Count the number of trainable parameters in model_bpm
        fearure_params_count = sum(p.numel() for p in model_feature.parameters() if p.requires_grad)  # Count the number of trainable parameters in model_feature
        classifier_params_count = sum(p.numel() for p in model_classifier.parameters() if p.requires_grad)  # Count the number of trainable parameters in model_classifier
        bpm_and_feature_params_count = bpm_params_count + fearure_params_count
        total_params_count = bpm_params_count + fearure_params_count + classifier_params_count
        wandb.config.update({'bpm_params_count': bpm_params_count,
                             'fearure_params_count': fearure_params_count,
                             'classifier_params_count': classifier_params_count,
                             'bpm_and_feature_params_count': bpm_and_feature_params_count,
                             'total_params_count': total_params_count,
                             })
        config = wandb.config  # access all hyperparameters through wandb.config, so logging matches execution!

        # ==== Loss, optimizer, scheduler ====
        loss_fn_1 = torch.nn.HuberLoss().to(device)
        loss_fn_2 = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(
            [{'name': 'lr_bpm', 'params': model_bpm.parameters(), 'lr': config.lr_bpm},
             {'name': 'lr_feature', 'params': model_feature.parameters(), 'lr': config.lr_feature},
             {'name': 'lr_class', 'params': model_classifier.parameters(), 'lr': config.lr_class},
             ], weight_decay=0.01,)  # 'AdamW' object has no attribute 'to', default weight_decay=0.01, ViT use 0.05
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.lr_bpm, config.lr_feature, config.lr_class],
            epochs=config.epochs,
            steps_per_epoch=data_info['train_batches'],)
            # pct_start=0.1, # default 0.3
            # anneal_strategy='linear', # default 'cos'
            # div_factor=8, # default 25
            # final_div_factor=256, # default 10000
            # )  # 'OneCycleLR' object has no attribute 'to'
        loss_ratio = torch.tensor(config.loss_ratio).to(device)  # avoid CPU to GPU transfer
        one = torch.tensor(1.0).to(device)

        # ==== Epoch loop start ====
        logging.debug(f'wandb config before starting epoch loop = {pformat(dict(config), sort_dicts=False)}')
        saved_checkpoint = [{'acc': 0.00, 'path': None},]  # Keep track of the saved checkpoint, so we can delete low accuracy ones when the list is too long
        for epoch in range(1, config.epochs+1):  # count epochs starting from 1
            time_start = time.time()
            logging.info(f"---------- Epoch: {epoch}----------")
            # ====================
            # ===== Training =====
            # ====================
            model_bpm.train()
            model_feature.train()
            model_classifier.train()
            train_loss_1_epoch = 0
            train_loss_2_epoch = 0
            train_loss_total_epoch = 0
            train_accuracy_epoch = 0
            # ===== Training batch loop start =====
            for batch, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch:02d} train", unit="batch"):
                image = data['image']  # Already in device
                feature = data['feature_finetune_vit']
                label = data['label']

                # Forward pass
                camera_pred = model_bpm(image)
                feature_pred = model_feature(camera_pred)
                logit_pred = model_classifier(feature_pred)
                label_pred = logit_pred.argmax(dim=1)

                # Compute loss
                loss_1 = loss_fn_1(feature_pred, feature)
                loss_2 = loss_fn_2(logit_pred, label)  # For Pytorch CrossEntropyLoss, the input is expected to contain raw. Do NOT apply softmax and argmax before passing it to the loss function.
                loss_total = loss_ratio * loss_1 + (one - loss_ratio) * loss_2
                train_accuracy = multiclass_accuracy(label_pred, target=label)

                loss_1_item = loss_1.item()  # Reduce GUP to CPU transfer, only call .item() one time
                loss_2_item = loss_2.item()
                loss_total_item = loss_total.item()
                train_accuracy_item = train_accuracy.item()

                train_loss_1_epoch += loss_1_item  # Both are already on cpu
                train_loss_2_epoch += loss_2_item
                train_loss_total_epoch += loss_total_item
                train_accuracy_epoch += train_accuracy_item

                # Backward pass
                optimizer.zero_grad()
                loss_total.backward()  # Compute gradients

                # Step with optimizer
                optimizer.step()
                scheduler.step()  # For OneCycleLR, step() should be invoked after each batch instead of after each epoch

                if batch % 75 == 0:
                    wandb.log({"every_75_train_batches/loss_1": loss_1_item,
                               "every_75_train_batches/loss_2": loss_2_item,
                               "every_75_train_batches/loss_total": loss_total_item,
                               "every_75_train_batches/accuracy": train_accuracy_item,
                               })

            # ===== Training batch loop end =====
            train_loss_1_epoch /= len(train_loader)
            train_loss_2_epoch /= len(train_loader)
            train_loss_total_epoch /= len(train_loader)
            train_accuracy_epoch /= len(train_loader)
            logging.info(f"TRAIN acc {train_accuracy_epoch:4f}, l_1 {train_loss_1_epoch:.5f}, l_2 {train_loss_2_epoch:.5f}, l_tot {train_loss_total_epoch:.5f}")
            wandb.log({"x-axis/epoch": epoch,
                       "train/loss_1": train_loss_1_epoch,
                       "train/loss_2": train_loss_2_epoch,
                       "train/loss_total": train_loss_total_epoch,
                       "train/accuracy": train_accuracy_epoch,
                       })
            # Log also the parameters histogram, gradient histogram, and learning rate
            for name, param in model_bpm.named_parameters():
                wandb.log({"x-axis/epoch": epoch,
                           f"parameters/model_bpm.{name}": wandb.Histogram(param.data.detach().cpu().numpy()),  # bins defult 64. https://docs.wandb.ai/ref/python/data-types/histogram/
                           f"gradient/model_bpm.{name}": wandb.Histogram(param.grad.data.detach().cpu().numpy()),  # histogram is not implemented for CUDA tensors
                           })
            for name, param in model_feature.named_parameters():
                wandb.log({"x-axis/epoch": epoch,
                           f"parameters/model_feature.{name}": wandb.Histogram(param.data.detach().cpu().numpy()),
                           f"gradient/model_feature.{name}": wandb.Histogram(param.grad.data.detach().cpu().numpy()),
                           })
            for name, param in model_classifier.named_parameters():
                wandb.log({"x-axis/epoch": epoch,
                           f"parameters/model_classifier.{name}": wandb.Histogram(param.data.detach().cpu().numpy()),
                           f"gradient/model_classifier.{name}": wandb.Histogram(param.grad.data.detach().cpu().numpy()),
                           })
            for param_group in optimizer.param_groups:
                wandb.log({f"x-axis/epoch": epoch, f"learning_rate/{param_group['name']}": param_group['lr']})  # Not on GPU, no need to transfer to CPU

            # ===================
            # ===== Testing =====
            # ===================
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
                for batch, data in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch:02d} test", unit="batch"):
                    image = data['image']  # Already in device
                    feature = data['feature_finetune_vit']
                    label = data['label']

                    # Forward pass
                    camera_pred = model_bpm(image)
                    feature_pred = model_feature(camera_pred)
                    logit_pred = model_classifier(feature_pred)
                    label_pred = logit_pred.argmax(dim=1)

                    # Compute loss
                    loss_1 = loss_fn_1(feature_pred, feature)
                    loss_2 = loss_fn_2(logit_pred, label)  # For Pytorch CrossEntropyLoss, the input is expected to contain raw. Do NOT apply softmax and argmax before passing it to the loss function.
                    loss_total = loss_ratio * loss_1 + (one - loss_ratio) * loss_2
                    test_accuracy = multiclass_accuracy(label_pred, target=label)

                    test_loss_1_epoch += loss_1.item()  # Ony onec, no need to reduce GUP to CPU transfer, aka no need to loss_item = loss.item()
                    test_loss_2_epoch += loss_2.item()
                    test_loss_total_epoch += loss_total.item()
                    test_accuracy_epoch += test_accuracy.item()

            # ===== Testing batch loop end =====
            test_loss_1_epoch /= len(test_loader)
            test_loss_2_epoch /= len(test_loader)
            test_loss_total_epoch /= len(test_loader)
            test_accuracy_epoch /= len(test_loader)
            runtime_minute = (time.time() - time_start) / 60.0
            wandb.log({"x-axis/epoch": epoch,
                       "test/loss_1": test_loss_1_epoch,
                       "test/loss_2": test_loss_2_epoch,
                       "test/loss_total": test_loss_total_epoch,
                       "test/accuracy": test_accuracy_epoch,
                       "runtime/minute": runtime_minute,
                       })
            logging.info(f"TEST acc {test_accuracy_epoch:4f}, l_1 {test_loss_1_epoch:.5f}, l_2 {test_loss_2_epoch:.5f}, l_tot {test_loss_total_epoch:.5f}")
            logging.info(f"EPOCH takes {runtime_minute:.2f} minutes to run")

            # ===== Save model checkpoint =====
            checkpoint_dir = script_path.parent / 'checkpoint' / f'{config.wanb_runname}'
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            checkpoint_file_dir = checkpoint_dir / f'acc{test_accuracy_epoch:.5f}-epoch{epoch:03d}-checkpoint.pth'
            # If this accuracy >= the max value of all current 'acc' in saved_checkpoint dictions, save this one
            if test_accuracy_epoch >= max([x['acc'] for x in saved_checkpoint]):
                # Save model state dict
                total_model_state_dict = {
                    'model_bpm.state_dict': model_bpm.state_dict(),
                    'model_feature.state_dict': model_feature.state_dict(),
                    'model_classifier.state_dict': model_classifier.state_dict(),
                    'wandb_config': dict(config),  # Save the config used in this run
                    'epoch': epoch,
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


# ==== Wandb login and start the sweep agent ====
os.environ["WANDB_API_KEY"] = 'c08ff557ed402c26a0c57ca0e7803529bdba9268'  # This is Leo's API key for the project
wandb.login()
wandb.agent(full_sweep_id, called_by_wandb_sweep_agent, count=1)
wandb.finish()
logging.info("wandb.finish() completed!")
