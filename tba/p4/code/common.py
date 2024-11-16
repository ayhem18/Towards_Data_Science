import torch
import torchvision
import os
import torchvision.transforms as tr

from pathlib import Path
from typing import List, Union

import torch, os, pickle, random
import pandas as pd, numpy as np 

from pathlib import Path
from torch.utils.data import DataLoader, Dataset


from functools import partial
from tqdm import tqdm
from typing import Union, Optional, Callable, List
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from torch.utils.tensorboard import SummaryWriter


from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR, LinearLR


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')

def seed_everything(seed: int = 69):
    # let's set reproducility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.use_deterministic_algorithms(True, warn_only=True) # certain layers have no deterministic implementation... we'll need to compromise on this one...
    torch.backends.cudnn.benchmark = False 

    # the final step to ensure reproducibility is to set the environment variable: # CUBLAS_WORKSPACE_CONFIG=:16:8
    import warnings
    # first check if the CUBLAS_WORSKPACE_CONFIG variable is set or not
    env_var = os.getenv('CUBLAS_WORKSPACE_CONFIG')
    if env_var is None:
        # the env variable was not set previously
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' # this is not the only viable value. I cannot remember the other value at the time of writing this code.
    else:
        if env_var not in [':16:8']:
            warnings.warn(message=f"the env variable 'CUBLAS_WORKSPACE_CONFIG' is set to the value {env_var}. setting it to: ':16:8' ")
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'        


def set_worker_seed(_, seed: int = 69):
    np.random.seed(seed=seed)
    random.seed(seed)


def load_data(parent_dir: Union[str, Path], augs: List):
    splits = ['train', 'val', 'test']
    train, val, test = [os.path.join(parent_dir, s) for s in splits]

    for s in [train, val, test]:
        os.makedirs(s, exist_ok=True)

    return [torchvision.datasets.Flowers102(root=d, 
                                            split=s, 
                                            transform=tr.Compose(augs), 
                                            download=True) 
                for s, d in zip(splits, [train, val, test])]
    

def train_model(
        train_ds: Dataset,
        test_ds: Dataset,
        net: torch.nn.Module,
        num_epochs: int,
        model_name: str,
        save_model_path: Optional[Union[str, Path]]=None,
        initial_lr:float=0.001,
        learning_scheduler_cls:Callable=None,
        learning_scheduler_kwargs=None,
        num_warmup_epochs: int=None
        ):
    
    seed_everything(seed=0)

    # set the dataloaders
    train_dl = DataLoader(train_ds, 
                          batch_size=256, 
                          shuffle=True, 
                          drop_last=True, 
                          worker_init_fn=partial(set_worker_seed, seed=0)
                          )
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, drop_last=False, worker_init_fn=partial(set_worker_seed, seed=0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    optimizer = SGD(params=net.parameters(), lr=initial_lr,)
    loss = torch.nn.CrossEntropyLoss()

    if learning_scheduler_cls is not None:
        learning_scheduler = learning_scheduler_cls(optimizer=optimizer, **learning_scheduler_kwargs)

    if num_warmup_epochs is not None:
        learning_scheduler = set_warmup_epochs(optimizer, main_lr_scheduler=learning_scheduler, num_warmup_epochs=num_warmup_epochs)

    writer = SummaryWriter()
    train_metrics = {}
    train_global_step = 0
    val_global_step = 0

    for epoch_index in tqdm(range(num_epochs)):
        # set to train mode``
        net.train()
        epoch_loss = 0

        train_correct_samples = 0
        train_total_samples = 0 # the size of the dataset is not necessarily the same as the total number of samples encountered during training (if drop_last = True for the dataloader)


        for x, y in tqdm(train_dl, desc=f'epoch: {epoch_index + 1} iterating through train dataset'):
            x, y = x.to(device), y.to(device) 
            optimizer.zero_grad()
            # forward pass
            y_model = net.forward(x).squeeze()
        
            loss_obj = loss.forward(y_model, y)
        
            epoch_loss += loss_obj.item()

            train_correct_samples += torch.sum((torch.argmax(y_model, dim=-1) == y).to(torch.float32)).item() 
            train_total_samples += len(y)

            # calculate gradients
            loss_obj.backward()

            # backward pass
            optimizer.step()

            writer.add_scalar(tag="train_batch_loss", scalar_value=round(loss_obj.item(), 4), global_step=train_global_step)
            train_global_step += 1

        if learning_scheduler is not None:
            learning_scheduler.step()

        epoch_loss /= len(train_dl)
        train_acc = train_correct_samples / train_total_samples
        train_metrics[f"train_loss_epoch_{epoch_index + 1}"] = round(epoch_loss, 4)
        train_metrics[f"train_accuracy_{epoch_index + 1}"] = round(train_acc, 4)

        writer.add_scalar(tag="train_epoch_loss", scalar_value=round(loss_obj.item(), 4), global_step=epoch_index)
        print(f"epoch: {epoch_index + 1}: train loss: {round(epoch_loss,4)}", end="\n")
        print(f"epoch: {epoch_index + 1}: train accuracy: {round(train_acc, 4) * 100} %", end="\n")


        val_epoch_loss = 0
        val_correct_samples = 0
        val_total_samples = 0

        # set to eval model
        net.eval()
        with torch.no_grad():
            for x, y in tqdm(test_dl, desc=f'epoch: {epoch_index + 1} iterating through val dataset'):
                x, y = x.to(device), y.to(device) 
                # forward pass
                y_model = net.forward(x).squeeze()
            
                loss_obj = loss.forward(y_model, y)
            
                val_epoch_loss += loss_obj.item()
            
                val_correct_samples += torch.sum((torch.argmax(y_model, dim=-1) == y).to(torch.float32)).item() 
                val_total_samples += len(y)

                # log the validation loss
                writer.add_scalar(tag="val_batch_loss", scalar_value=loss_obj.item(), global_step=val_global_step)
                val_global_step += 1

        val_epoch_loss /= len(test_dl)
        val_acc = val_correct_samples / val_total_samples
        train_metrics[f"val_loss_epoch_ {epoch_index + 1}"] = round(val_epoch_loss, 4)
        train_metrics[f"val_accuracy_epoch {epoch_index + 1}"] = round(val_acc,4)

        # log it to TensorBoard
        writer.add_scalar(tag="val_epoch_loss", scalar_value=val_epoch_loss, global_step=epoch_index)

        print(f"epoch: {epoch_index + 1}: val loss: {round(val_epoch_loss, 4)}", end="\n")
        print(f"epoch: {epoch_index + 1}: val accuracy: {round(val_acc, 4) * 100} %", end="\n")

    if save_model_path is None:
        save_model_path = os.path.join(DATA_FOLDER, 'models', model_name)        
        os.makedirs(save_model_path, exist_ok=True)
    
    save_model_path = os.path.join(save_model_path, f'{model_name}.pt')

    torch.save(net.state_dict(), save_model_path)
    
    return net
    



def set_warmup_epochs(optimizer: Optimizer,
                    main_lr_scheduler: LRScheduler, 
                    num_warmup_epochs: int) -> SequentialLR:

    warmup_lr_scheduler = LinearLR(optimizer=optimizer, 
                        total_iters=num_warmup_epochs, 
                        start_factor=0.01,
                        end_factor=1,
                        )

    final_lr_scheduler = SequentialLR(optimizer=optimizer, 
                                schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                                milestones=[num_warmup_epochs])

    return final_lr_scheduler