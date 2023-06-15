import torch
import numpy as np 
from torch import nn
from math import ceil

from collections import OrderedDict
from pytorch_utilities import dimensions_block


class my_vgg(nn.Module):

    last_layer_min_units = 32

    @classmethod
    def last_FC_units(cls, num_classes: int):
        log2 = np.log2(num_classes)
        return max(cls.last_layer_min_units, 4 *  2 ** ceil(log2))



    def __init__(self, input_shape: tuple[int, int, int], num_classes: int, num_layers: int = 6, initial_num_filters: int = 32, logits: bool = True,
                 *args, **kwargs) -> None:
        # first call the constructer of the super class 
        super().__init__(*args, **kwargs)        

        # extract the dimensions on the input
        h, w, c = input_shape # width , height, channels

        # first make sure the number of layers is correct
        assert num_layers in [4, 6, 8]

        # make sure to consider the case of binary classification
        self.output_shape = num_classes if num_classes > 2 else 1
        fc_units = self.last_FC_units(num_classes)

        # this is important for training and post processing
        self.logits = logits

        # 1st convolutional block
        self.conv_block_1 = nn.Sequential(
            # the first convolutional applied 'same' padding
            nn.Conv2d(in_channels=c,
                      out_channels=initial_num_filters, 
                      kernel_size=3, 
                      stride=1,
                      padding='same'), 
            nn.ReLU(),

            # no padding is applied in the 2nd
            nn.Conv2d(in_channels=initial_num_filters,
                      out_channels=2 * initial_num_filters, 
                      kernel_size=3, 
                      stride=2,
                      padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, 
                        stride=2)
        )

        # 2nd convolutional block
        self.conv_block_2 = nn.Sequential(
            # the first convolutional applied 'same' padding
            nn.Conv2d(in_channels=2 * initial_num_filters,
                      out_channels=2 * initial_num_filters, 
                      kernel_size=3, 
                      stride=1,
                      padding='same'), 
            nn.ReLU(),

            # no padding is applied in the 2nd
            nn.Conv2d(in_channels=2 * initial_num_filters,
                      out_channels=4 * initial_num_filters, 
                      kernel_size=3, 
                      stride=2,
                      padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, 
                        stride=2)
        )

        # the 3rd convolutional block
        self.conv_block_3 = nn.Sequential(
            # the first convolutional applied 'same' padding
            nn.Conv2d(in_channels=4 * initial_num_filters,
                      out_channels= 4 * initial_num_filters, 
                      kernel_size=3, 
                      stride=1,
                      padding='same'), 
            nn.ReLU(),

            # no padding is applied in the 2nd
            nn.Conv2d(in_channels=4 * initial_num_filters,
                      out_channels=8 * initial_num_filters, 
                      kernel_size=3, 
                      stride=2,
                      padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, 
                        stride=2)
        )


        # since the architecture of the network is parametric, we need to define an additional attribute to use in the forward function call
        block_indices = OrderedDict({4: self.conv_block_1, 6: self.conv_block_2, 8: self.conv_block_3})
        self.conv = nn.ModuleList([block for i, block in block_indices.items() if num_layers >= i])

        
        for module in self.conv:
            w, h = dimensions_block(module)
        
        classifier_input = w * h * self.conv[-1].out_channels

        # time for the classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            # the first layer 
            nn.Linear(in_features=classifier_input,out_features=fc_units), 
            nn.ReLU(),
            nn.Linear(in_features=fc_units, out_features=self.output_shape)            
        )     

        if not logits:
            final_layer = nn.Softmax(dim=-1) if self.output_shape > 1 else nn.Sigmoid()
            
            # add an additional layer if logist argument is set to False
            self.classifier.append(final_layer)


        def forward(x: torch.tensor):
            
            for module in self.conv:
                x = module(x)
            
            return self.classifier(x)



        
