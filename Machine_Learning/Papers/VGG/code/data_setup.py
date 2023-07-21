"""
This script is written to prepare the Dogs VS cats dataset for modeling.
"""

import os
from pathlib import Path
from typing import Union
from pytorch_modular.directories_and_files import copy_directories, unzip_data_file

HOME = os.getcwd()


def prepare_DVC_dataset(dvc_path: Union[str, Path] = None):
    if dvc_path is None:
        dvc_path = os.path.join(HOME, 'dogs-vs-cats.zip')

    directory = unzip_data_file(dvc_path)

    assert {'test1', 'train'}.issubset(set(os.listdir(directory)))

    # rename 'test1' to 'test'
    os.rename(os.path.join(directory, 'test1'), os.path.join(directory, 'test'))

    train_dir = os.path.join(directory, 'train')
    cat_dir, dog_dir = os.path.join(train_dir, 'cat'), os.path.join(train_dir, 'dog')
    # create a folder for dogs and cats within the train directory
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    # move all files with the string dog in their names to the dog folder
    copy_directories(train_dir, dog_dir, copy=False, filter_directories=lambda x: 'dog' in x)
    copy_directories(train_dir, cat_dir, copy=False, filter_directories=lambda x: 'cat' in x)


if __name__ == '__main__':
    prepare_DVC_dataset()
