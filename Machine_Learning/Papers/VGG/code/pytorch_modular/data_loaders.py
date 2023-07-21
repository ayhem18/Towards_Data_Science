"""
This script contains functionalities to create data loaders for different Computer Vision tasks

PS: Currently only image classification functionality is implemented
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from copy import copy
from .helper_functions import reverse_dict
from typing import Union
from pathlib import Path
from functools import partial

NUM_WORKERS = os.cpu_count() // 2


# let's define a function that would help extracting only a small number of files from a path
def extract_n_files(path: str, limit: int):
    i = -1
    while limit is not None:
        yield i < limit
        i += 1

    return True


def create_dataloaders(
        train_dir: Union[str, Path],
        train_transform: transforms.Compose,
        batch_size: int,
        test_dir: Union[str, Path] = None,
        test_transform: transforms.Compose = None,
        num_workers: int = NUM_WORKERS,
        val_dir: Union[str, Path] = None,
        val_transform: transforms.Compose = None,
        # collate_function variables
        collate_function_train: callable = None,
        collate_function_val: callable = None,
        collate_function_test: callable = None,
        max_num_images_train: int = None,
        max_num_images_val: int = None,
        max_num_images_test: int = None
) -> Union[tuple[DataLoader, dict],
           tuple[DataLoader, DataLoader, dict],
           tuple[DataLoader, DataLoader, DataLoader, dict]]:

    # if the test_transform is set to None, set it to the train_transform
    if test_transform is None:
        test_transform = copy(train_transform)

    if val_transform is None:
        val_transform = copy(train_transform)

    # let's define functions that will help limit the number of files
    is_valid_file_train = partial(extract_n_files, limit=max_num_images_train)
    is_valid_file_val = partial(extract_n_files, limit=max_num_images_val)
    is_valid_file_test = partial(extract_n_files, limit=max_num_images_test)

    # create the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform, is_valid_file=is_valid_file_train)

    # as the validation dataset, may or may not be present
    val_dataloader, test_dataloader = None, None

    # the default value of the collate_fn argument is None. Therefore, we can pass it directly
    # without further checking

    if val_dir is not None:
        # create the dataset object
        val_data = datasets.ImageFolder(val_dir, transform=val_transform, is_valid_file=is_valid_file_val)
        # create the corresponding dataLoader
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_function_val
        )

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_function_train
    )

    if test_dir is not None:
        test_data = datasets.ImageFolder(test_dir, transform=test_transform, is_valid_file=is_valid_file_test)

        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_function_test
        )

    output = (train_dataloader,)

    if val_dataloader is not None:
        output += (val_dataloader,)

    if test_dataloader is not None:
        output += (test_dataloader,)

    output += (reverse_dict(train_data.class_to_idx),)

    return output
