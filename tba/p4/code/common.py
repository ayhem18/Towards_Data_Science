import torch
import torchvision
import os
import torchvision.transforms as tr

from pathlib import Path
from typing import List, Union

import torch, os, pickle, random
import pandas as pd, numpy as np 

from pathlib import Path
from typing import List, Iterator, Union, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

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



from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR, LinearLR


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