import torch, os
import torchvision.transforms as tr

from pathlib import Path
from typing import List, Union

from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional, List
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR, LinearLR

from mypt.code_utilities.pytorch_utilities import seed_everything, get_default_device
from mypt.data.dataloaders.standard_dataloaders import initialize_val_dataloader, initialize_train_dataloader

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')



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

def train_model(
        train_ds: Dataset,
        test_ds: Dataset,
        train_batch_size:int, 
        test_batch_size:int,
        net: torch.nn.Module,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        learning_scheduler: torch.optim.lr_scheduler.LRScheduler, 
        num_epochs: int,
        num_warmup_epochs: int=None,
        save_model_path: Optional[Union[str, Path]]=None,
        log_dir: Optional[Union[str, Path]]=None
        ):
    
    seed_everything(seed=0)

    # set the dataloaders    
    train_dl = initialize_train_dataloader(train_ds, seed=0, batch_size=train_batch_size, num_workers=2)
    test_dl = initialize_val_dataloader(test_ds, seed=0, batch_size=test_batch_size, num_workers=2)

    device = get_default_device()
    net = net.to(device)

    # optimizer = SGD(params=net.parameters(), lr=initial_lr,)
    loss = torch.nn.CrossEntropyLoss()
    
    # add warnmup epochs
    if num_warmup_epochs is not None:
        learning_scheduler = set_warmup_epochs(optimizer, main_lr_scheduler=learning_scheduler, num_warmup_epochs=num_warmup_epochs)

    if log_dir is None:
        log_dir = os.path.join(SCRIPT_DIR, 'logs', model_name)

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
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
        print(f"epoch: {epoch_index + 1}: train accuracy: {round(train_acc * 100, 4) } %", end="\n")


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
        print(f"epoch: {epoch_index + 1}: val accuracy: {round(val_acc * 100, 4) } %", end="\n")

    if save_model_path is None:
        save_model_path = os.path.join(DATA_FOLDER, 'models', model_name)        

    os.makedirs(save_model_path, exist_ok=True)
    save_model_path = os.path.join(save_model_path, f'{model_name}.pt')

    torch.save(net.state_dict(), save_model_path)
    
    return net

def calculate_accuracy(model: torch.nn.Module, 
                       ds: Dataset, 
                       batch_size: int ) -> float:

    dl = initialize_val_dataloader(ds, seed=0, batch_size=batch_size, num_workers=2)

    device = get_default_device()
    model = model.to(device)

    correct_samples = 0
    total_samples = 0

    # set to eval model
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device) 
            # forward pass
            y_model = model.forward(x).squeeze()        
            correct_samples += torch.sum((torch.argmax(y_model, dim=-1) == y).to(torch.float32)).item() 
            total_samples += len(x)

    return round(correct_samples / total_samples, 4)

