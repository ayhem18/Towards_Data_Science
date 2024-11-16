"""
This script contains the implementations of the different Convolutional Neural Networks
"""


import torch, os, pickle, random
import pandas as pd, numpy as np 
import torchvision.transforms as tr

from functools import partial
from tqdm import tqdm
from typing import List, Iterator, Union, Optional, Tuple
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter

from common import predict, seed_everything, set_worker_seed, load_data, set_warmup_epochs

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


class ImprovedCnn(torch.nn.Module):
    # this class will be improved using 3 techinuqes: 
    # 1. batch normalization
    # 2. warm up epochs + learning rate scheduler
    # 3. boosting
    
    def __init__(self):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, 
                            out_channels=32,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(in_channels=32, 
                            out_channels=64,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(in_channels=64, 
                            out_channels=128,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            torch.nn.BatchNorm2d(128)
        )

        self.fc = torch.nn.Sequential(torch.nn.Flatten(), 
                                      torch.nn.LazyLinear(out_features=512), 
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(512, 102))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc.forward(self.conv_block.forward(x))
    

    def to(self, *args, **kwargs):
        self.conv_block = self.conv_block.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        return self
    


def train(
        train_ds: Dataset,
        test_ds: Dataset,
        net: torch.nn.Module,
        num_epochs: int,
        num_warmup_epochs: int,
        save_model_path: Optional[Union[str, Path]]=None
        ):
    seed_everything(seed=0)

    # set the dataloaders
    train_dl = DataLoader(train_ds, 
                          batch_size=512, 
                          shuffle=True, 
                          drop_last=True, 
                          worker_init_fn=partial(set_worker_seed, seed=0)
                          )
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, drop_last=False, worker_init_fn=partial(set_worker_seed, seed=0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'
    
    net = net.to(device)
    


    optimizer = SGD(params=net.parameters(), lr=0.001,)
    loss = torch.nn.CrossEntropyLoss()

    # a write used for TensorBoard
    writer = SummaryWriter()

    main_lrs = ExponentialLR(optimizer=optimizer, gamma=0.99)

    lr_scheduler = set_warmup_epochs(optimizer, main_lrs, num_warmup_epochs)

    # lr_scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.01, total_iters=num_epochs) 

    train_metrics = {}

    for i in tqdm(range(num_epochs)):
        # set to train mode
        net.train()
        epoch_loss = 0
        train_acc = 0

        for x, y in tqdm(train_dl, desc=f'epoch: {i + 1} iterating through train dataset'):
            x, y = x.to(device), y.to(device) 
            optimizer.zero_grad()
            # forward pass
            y_model = net.forward(x).squeeze()
        
            loss_obj = loss.forward(y_model, y)
        
            epoch_loss += loss_obj.item()

            train_acc += torch.mean((torch.argmax(y_model) == y).to(torch.float32)) 

            # calculate gradients
            loss_obj.backward()

            # backward pass
            optimizer.step()

            writer.add_scalar(tag="train_batch_loss", scalar_value=loss_obj.item())

        lr_scheduler.step() 

        epoch_loss /= len(train_dl)
        print(f"epoch: {i + 1}: train loss: {epoch_loss}")
        train_metrics[f"train_loss_epoch_{i + 1}"] = epoch_loss

        print(f"epoch: {i + 1}: train loss: {epoch_loss}")
        train_metrics[f"train_accuracy_{i + 1}"] = train_acc / len(train_dl) 


        val_epoch_loss = 0
        # set to eval model
        net.eval()
        with torch.no_grad():
            for x, y in tqdm(test_dl, desc=f'epoch: {i + 1} iterating through val dataset'):
                x, y = x.to(device), y.to(device) 
                # forward pass
                y_model = net.forward(x).squeeze()
            
                loss_obj = loss.forward(y_model, y)
            
                val_epoch_loss += loss_obj.item()
                # log the validation loss
                writer.add_scalar(tag="val_batch_loss", scalar_value=loss_obj.item())


        val_epoch_loss /= len(test_dl)
        print(f"epoch: {i + 1}: val loss: {val_epoch_loss}")
        train_metrics[f"val_loss_epoch_ {i + 1}"] = val_epoch_loss

        # log it to TensorBoard
        writer.add_scalar(tag="val_epoch_loss", scalar_value=val_epoch_loss)


    model_name = 'ann'
    if save_model_path is None:
        save_dir = os.path.join(DATA_FOLDER, 'models', model_name)        
        os.makedirs(save_dir, exist_ok=True)
        save_model_path = os.path.join(save_dir, f'{model_name}.pt')


    return net



if __name__ == '__main__':
    bcnn = ImprovedCnn()

    # add mini-max scaling 
    test_ds, val_ds, train_ds = load_data(parent_dir=DATA_FOLDER, augs=[tr.Resize(size=(200, 200)), # resize to the same input shape
                                                                        tr.ToTensor(), # convert to a tensor 
                                                                        ]) 


    print(len(train_ds), len(val_ds), len(test_ds))

    # train(train_ds, 
    #       val_ds, 
    #       net = bcnn, 
    #       num_epochs=20, 
    #       num_warmup_epochs=5, 
    #       save_model_path=os.path.join(DATA_FOLDER, 'models', 'bcnn'))
