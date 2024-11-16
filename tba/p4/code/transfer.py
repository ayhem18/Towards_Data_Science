import torch, os
import torchvision.transforms as tr

from pathlib import Path
from torch import nn
from typing import Iterator, Tuple, Optional, Union

from torch.optim.lr_scheduler import ExponentialLR

from mypt.backbones import resnetFeatureExtractor as rfe
from mypt.linearBlocks import fully_connected_blocks as fcb
from mypt.dimensions_analysis import dimension_analyser as da


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


class ResnetClassifier(nn.Module):
    def __init__(self,
                input_shape: Tuple[int, int, int],
                num_classes: int,
                num_fc_layers: int,
                dropout: Optional[float] = None,
                fe_num_blocks: int=-1, # use all the layer blocks of the Resnet feature extractor
                architecture: int = 50, # use Resnet50
                freeze: Union[int, bool]=False, # do not freeze any of the  layers of the pretrained model, 
                freeze_layers: bool=True) -> None:

        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(f"Make sure the input_shape argument represents the sample input shape; 3 dimensions !! Found: {input_shape}")
 
        # the feature extractor or the encoder "f" as denoted in the paper.
        self.fe = rfe.ResNetFeatureExtractor(num_layers=fe_num_blocks, 
                                        architecture=architecture,
                                        freeze=freeze, 
                                        freeze_layers=freeze_layers)
        self.flatten_layer = nn.Flatten()

        dim_analyser = da.DimensionsAnalyser(method='static')

        # calculate the number of features passed to the classification head.
        _, in_features = dim_analyser.analyse_dimensions(input_shape=(10,) + input_shape, net=nn.Sequential(self.fe, nn.Flatten()))

        # calculate the output of the
        self.fc = fcb.ExponentialFCBlock(output=1 if num_classes == 2 else num_classes, 
                                        in_features=in_features, 
                                        num_layers=num_fc_layers, 
                                        dropout=dropout)
        
        self.model = nn.Sequential(self.fe, self.flatten_layer, self.fc)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(x)


    def __str__(self):
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__() 
    
    def children(self) -> Iterator[nn.Module]:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.model
        return self.model.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.model.named_children()

    def to(self, *args, **kwargs):
        # self.model = self.model.to(*args, **kwargs)
        self.fe = self.fe.to(*args, **kwargs)
        self.flatten_layer = self.flatten_layer.to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        return self 

    def __call__(self, x: torch.Tensor):
        return self.forward(x)



from torch.optim.sgd import SGD

if __name__ == '__main__':
    from common import train_model, load_data

    # add mini-max scaling 
    train_ds, val_ds, test_ds = load_data(parent_dir=DATA_FOLDER, augs=[tr.ToTensor(), # convert to a tensor 
                                                                    tr.Resize(size=(200, 200)), # resize to the same input shape
                                                                    tr.Lambda(lambda x: x / 255.0) # min
                                                                    ]) 

    resnet = ResnetClassifier(input_shape=(3, 200, 200), 
                              num_classes=102, 
                              num_fc_layers=3, # add 3 fully connected layers 
                              dropout=0.1, # apply 0.1 dropout at each fc layer
                              fe_num_blocks=-1, # use the entire resnet model (the class has the ability to choose parts of the model)
                              freeze=4, # freeze 4 out 5 residual blocks
                              architecture=50 # Resnet50 should do the trick
                              )


    optimizer = SGD(params=resnet.parameters(), lr=0.01,)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

    train_model(
            train_ds, 
            val_ds, 
            train_batch_size=256, 
            test_batch_size=512,
            model_name="resnet",
            net = resnet, 
            optimizer=optimizer,
            learning_scheduler=lr_scheduler,
            num_epochs=20, 
            num_warmup_epochs=5, 
            save_model_path=os.path.join(DATA_FOLDER, 'models', 'resnet'),
            )
