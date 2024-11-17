"""
This script contains the implementations of the different Convolutional Neural Networks
"""


import torch, os
from pathlib import Path
from torch.optim.sgd import SGD

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


class BaselineCnn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, 
                            out_channels=32,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            torch.nn.Conv2d(in_channels=32, 
                            out_channels=64,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            torch.nn.Conv2d(in_channels=64, 
                            out_channels=128,
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
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

    bcnn = BaselineCnn()

    # add mini-max scaling 
    train_ds, val_ds, test_ds = load_data(parent_dir=DATA_FOLDER, augs=[]) 

    optimizer = SGD(params=bcnn.parameters(), lr=0.001,)

    train_model(
            train_ds, 
            val_ds, 
            train_batch_size=256, 
            test_batch_size=512,
            net = bcnn, 
            model_name="baseline_cnn",
            optimizer=optimizer,
            learning_scheduler=None, # no learning rate scheduling
            num_epochs=10, 
            num_warmup_epochs=None, # no warmup epochs 
            save_model_path=os.path.join(DATA_FOLDER, 'models', 'baseline_cnn'),
            )


