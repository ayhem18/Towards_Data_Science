import torch
from torch import nn
from typing import Union


def __dimensions_after_conv(h: int, w: int, conv: nn.Conv2d) -> tuple[int, int]:
    # this code is based on the documentation of conv2D module pytorch:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    # extract the numerical features first 
    s1, s2 = conv.stride
    k1, k2 = conv.kernel_size
    d1, d2 = conv.dilation

    # the padding is tricky
    if conv.padding == 'same':
        return h, w
    
    if conv.padding == 'valid':
        p1, p2 = 0, 0
    
    else:
        p1, p2 = conv.padding


    new_h =  int((h + 2 * p1 - d1 * (k1 - 1) - 1) / s1) + 1
    new_w =  int((w + 2 * p2 - d2 * (k2 - 1) - 1) / s2) + 1

    return new_h, new_w    

def __dimensions_after_pool(h: int, w: int, pool: Union[nn.MaxPool2D, nn.AvgPool2d]) -> tuple[int, int]:
    k1, k2 = pool.kernel_size
    s1, s2 = pool.stride
    return int((h - k1)/ s1) + 1, int((w - k2) / s2) + 1

def dimensions_block(input_shape: Union[tuple[int, int], int], block: nn.Sequential) -> tuple[int, int]:

    # first extract the initial input shape
    input_h, input_w = input_shape if  isinstance(input_shape, tuple) else (input_shape, input_shape)
    # iterate through the layers of the block and modify the shape accordingly depending on the layer
    for layer in block:
        if isinstance(layer, nn.Conv2d):
            input_h, input_w = __dimensions_after_conv(input_h, input_w, layer)
        elif isinstance(layer, nn.MaxPool2d, nn.AvgPool2d):
            input_h, input_w = __dimensions_after_pool(input_h, input_w, layer)
             
    return (input_h, input_w)


    
    