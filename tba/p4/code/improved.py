"""
This script contains the implementations of the different Convolutional Neural Networks
"""

import os, torch
import torchvision.transforms as tr
from pathlib import Path

from torch.optim.lr_scheduler import ExponentialLR


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


class ImprovedCnn(torch.nn.Module):
    # this class will be improved using 3 techinuqes: 
    # 1. batch normalization
    # 2. warm up epochs + learning rate scheduler
    # 3. increasing the model capacity: more convolutional layers + non-linear activation layers

    def __init__(self):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, 
                            out_channels=32,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding="same"),

            torch.nn.Conv2d(in_channels=32, 
                            out_channels=64,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding="same"),

            torch.nn.Conv2d(in_channels=64, 
                            out_channels=128,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),

            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=128, 
                            out_channels=128,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding="same"),

            torch.nn.Conv2d(in_channels=128, 
                            out_channels=128,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding="same"),

            torch.nn.Conv2d(in_channels=128, 
                            out_channels=256,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),

            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=256, 
                            out_channels=512,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding="same"),

            torch.nn.Conv2d(in_channels=512, 
                            out_channels=512,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding="same"),

            torch.nn.Conv2d(in_channels=512, 
                            out_channels=1024,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),

            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
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
    


if __name__ == '__main__':
    from common import train_model, load_data
    cnn = ImprovedCnn()

    # add mini-max scaling 
    test_ds, val_ds, train_ds = load_data(parent_dir=DATA_FOLDER, augs=[tr.Resize(size=(200, 200)), # resize to the same input shape
                                                                        tr.ToTensor(), # convert to a tensor 
                                                                        ]) 

    train_model(train_ds, 
          val_ds, 
          net = cnn, 
          num_epochs=20, 
          num_warmup_epochs=5, 
          save_model_path=os.path.join(DATA_FOLDER, 'models', 'bcnn'),
          model_name="improved_cnn",
          learning_scheduler_cls=ExponentialLR,
          learning_scheduler_kwargs={"gamma": 0.9},
          initial_lr= 0.01
          )
