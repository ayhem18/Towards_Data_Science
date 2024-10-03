import torch, os, pickle
import pandas as pd, numpy as np 

from tqdm import tqdm
from typing import List, Iterator, Union, Optional, Tuple
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

from models import evaluate_reg_model


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


class Network(torch.nn.Module): 
    def __init__(self, in_features: int, hidden_units: List[int], dropout:float):
        
        super().__init__()

        input_layer = torch.nn.Linear(in_features=in_features, out_features=hidden_units[0])
        
        layers = [input_layer, torch.nn.ReLU(), torch.nn.Dropout(dropout)]

        for i in range(len(hidden_units) - 1):
            layers.extend([                
                torch.nn.Linear(hidden_units[i], hidden_units[i + 1]),
                torch.nn.ReLU(), 
                torch.nn.Dropout(dropout)
            ])

        layers.append(torch.nn.Linear(hidden_units[-1], 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net.forward(x)

    def __str__(self) -> str:
        return self.net.__str__()

    def children(self) -> Iterator[torch.nn.Module]:
        return self.net.children()

    def named_children(self) -> Iterator[torch.nn.Module]:
        return self.net.children()

    def modules(self) -> Iterator[torch.nn.Module]:
        return self.net.modules()

    def to(self, *args, **kwargs):
        self.net = self.net.to(*args, **kwargs)
        return self


class DataFrameDs(Dataset):
    def __init__(self, 
                 X_train: Union[str, Path, pd.DataFrame],
                 y_train: Optional[Union[str, Path, pd.DataFrame]]=None,
                 y_label: str=None, 
                 ):
        if y_train is None and y_label is None:
            raise ValueError(f"either y_train or _ylabel must be passed")

        # read the dataframe
        if isinstance(X_train, (str, Path)):
            self.X_train = pd.read_csv(X_train)
        else:
            self.X_train = X_train

        if y_label is not None:
            self.y_train = self.X_train[y_label]
            self.X_train.pop(y_label)
        else:
            if isinstance(y_train, (str, Path)):
                self.y_train = pd.read_csv(y_train)
            else:
                self.y_train = y_train

        # convert to numpy array
        self.X_train = self.X_train.values
        self.y_train = self.y_train.values.squeeze()


    def __getitem__(self, index) -> Tuple[torch.Tensor, float]:
        # pytorch requires data with the float32 data type
        return torch.from_numpy(self.X_train[index, :],).to(torch.float32), self.y_train[index].astype('float32')

    def __len__(self) -> int:
        return len(self.X_train)


def train_ann(X_train: pd.DataFrame, 
              y_train: pd.DataFrame,
              X_test: pd.DataFrame,
              y_test: pd.DataFrame,
              ann_hidden_units: List[int],
              num_epochs: int,
              save_model_path: Optional[Union[str, Path]]=None
              ):

    net = Network(in_features=X_train.shape[1], 
                  hidden_units=ann_hidden_units, 
                  dropout=0.2)

    # dataset
    train_ds = DataFrameDs(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)

    test_ds = DataFrameDs(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=True, drop_last=False)

    optimizer = Adam(net.parameters(), lr=0.01, weight_decay=10**-3)

    loss = torch.nn.MSELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = net.to(device)

    train_metrics = {}

    for i in tqdm(range(num_epochs)):
        epoch_loss = 0

        for x, y in tqdm(train_dl, desc=f'epoch: {i + 1} iterating through dataset'):
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

        epoch_loss /= len(train_dl)
        print(f"epoch: {i + 1}: train loss: {epoch_loss}")

        train_metrics[i] = epoch_loss

    # set the network to the evaluation model
    net.eval()
    y_pred = None

    with torch.no_grad():
        for x, _ in test_dl:
            x = x.to(device)
            y_hat = net.forward(x).cpu().numpy()
            
            if y_hat.ndim == 1:
                y_hat = np.expand_dims(y_hat, axis=-1)

            if y_pred is None:
                y_pred = y_hat
            else:
                y_pred = np.concatenate([y_pred, y_hat], axis=0)

    y_pred = y_pred.squeeze()
    # compute the test_metrics
    test_metrics = evaluate_reg_model(x_test=X_test, y_test=y_test, y_pred=y_pred)

    model_name = 'ann'
    if save_model_path is None:
        save_dir = os.path.join(DATA_FOLDER, 'models', model_name)        
        os.makedirs(save_dir, exist_ok=True)
        save_model_path = os.path.join(save_dir, f'{model_name}.ob')

    train_metrics_save_path, test_metrics_save_path = (os.path.join(Path(save_model_path).parent, f'{model_name}_train_metrics.ob'), 
                                                        os.path.join(Path(save_model_path).parent, f'{model_name}_test_metrics.ob'))

    for obj, path in [(net, save_model_path), (train_metrics, train_metrics_save_path), (test_metrics, test_metrics_save_path)]:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    return net, train_metrics, test_metrics


if __name__ == '__main__':
    df_train = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv'))

    df_test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_FOLDER, 'y_test.csv'))

    net, train_metrics, test_metrics = train_ann(X_train=df_train, 
                                                 y_train=y_train, 
                                                 X_test=df_test, 
                                                 y_test=y_test,
                                                 ann_hidden_units=[128, 32, 8], 
                                                 num_epochs=10)
