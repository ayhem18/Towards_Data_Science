"""
This script contains general helper functions that are not particularly related to Pytorch
and Deep Learning in general, but used in other scripts
"""
from typing import Union
from collections import Counter


def reverse_dict(dictionary: Union[dict, Counter]) -> dict:
    return dict(zip(list(dictionary.values()), list(dictionary.keys())))


