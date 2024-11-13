import torch, os, pickle, random
import pandas as pd, numpy as np 
import torchvision.transforms as tr
from functools import partial
from tqdm import tqdm
from typing import Union, Optional
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from common import seed_everything, set_worker_seed, load_data
from cnn import BaselineCnn



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')



def predict(net: torch.nn.Module, test_ds: Dataset) -> np.ndarray:
    dl = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set the network to the evaluation model
    net.eval()

    y_pred = None
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            y_hat = net.forward(x).cpu().numpy()
            
            if y_hat.ndim == 1:
                y_hat = np.expand_dims(y_hat, axis=-1)

            if y_pred is None:
                y_pred = y_hat
            else:
                y_pred = np.concatenate([y_pred, y_hat], axis=0)

    y_pred = y_pred.squeeze()
    return y_pred



def train(
        train_ds: Dataset,
        test_ds: Dataset,
        net: torch.nn.Module,
        num_epochs: int,
        save_model_path: Optional[Union[str, Path]]=None
        ):
    seed_everything(seed=0)

    # set the dataloaders
    train_dl = DataLoader(train_ds, 
                          batch_size=128, 
                          shuffle=True, 
                          drop_last=True, 
                          worker_init_fn=partial(set_worker_seed, seed=0)
                          )
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=False, worker_init_fn=partial(set_worker_seed, seed=0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    optimizer = SGD(params=net.parameters(), lr=0.001,)
    loss = torch.nn.CrossEntropyLoss()

    # a write used for TensorBoard
    writer = SummaryWriter()

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

            # calculate gradients
            loss_obj.backward()

            # backward pass
            optimizer.step()

            writer.add_scalar(tar="train_batch_loss", scalar_value=loss_obj.item())

        # lr_scheduler.step() 

        epoch_loss /= len(train_dl)
        print(f"epoch: {i + 1}: train loss: {epoch_loss}")
        train_metrics[f"train_loss_epoch_{i + 1}"] = epoch_loss

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
                writer.add_scalar(tar="val_batch_loss", scalar_value=loss_obj.item())


        val_epoch_loss /= len(test_dl)
        print(f"epoch: {i + 1}: val loss: {val_epoch_loss}")
        train_metrics[f"val_loss_epoch_ {i + 1}"] = val_epoch_loss

        # log it to TensorBoard
        writer.add_scalar(tag="val_epoch_loss", scalar_value=val_epoch_loss)

    

    # # predict on the train dataset
    # y_train_pred = predict(net, X_train, y_train)
    # y_test_pred = predict(net, X_test, y_test)

    # # compute the test_metrics
    # train_metrics.update(evaluate_reg_model(X_train, y_train, y_pred=y_train_pred))
    # test_metrics = evaluate_reg_model(x_test=X_test, y_test=y_test, y_pred=y_test_pred)

    model_name = 'ann'
    if save_model_path is None:
        save_dir = os.path.join(DATA_FOLDER, 'models', model_name)        
        os.makedirs(save_dir, exist_ok=True)
        save_model_path = os.path.join(save_dir, f'{model_name}.pt')

    # train_metrics_save_path, test_metrics_save_path = (os.path.join(Path(save_model_path).parent, f'{model_name}_train_metrics.ob'), 
    #                                                     os.path.join(Path(save_model_path).parent, f'{model_name}_test_metrics.ob'))

    # for obj, path in [(net, save_model_path), (train_metrics, train_metrics_save_path), (test_metrics, test_metrics_save_path)]:
    #     if isinstance(obj, torch.nn.Module):
    #         torch.save(net.state_dict(), path)
    #     else:
    #         with open(path, 'wb') as f:
    #             pickle.dump(obj, f)

    # return net, train_metrics, test_metrics

    return net



if __name__ == '__main__':
    bcnn = BaselineCnn()

    # add mini-max scaling 
    train_ds, val_ds, test_ds = load_data(parent_dir=DATA_FOLDER, augs=[tr.ToTensor(), tr.Resize(size=(200, 200))]) 

    train(train_ds, val_ds, net = bcnn, num_epochs=5, save_model_path=os.path.join(DATA_FOLDER, 'models', 'bcnn'))