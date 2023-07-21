"""
This scripts contains functionalities to manipulate files and directories
"""

import os
from pathlib import Path
from typing import Union
import zipfile
import shutil

HOME = os.getcwd()


def abs_path(path: Union[str, Path]) -> Path:
    return Path(path) if os.path.isabs(path) else os.path.join(HOME, path)


def squeeze_directory(directory_path: Union[str, Path]) -> None:
    # Given a directory with only one subdirectory, this function moves all the content of
    # subdirectory to the parent directory

    # first convert to abs
    path = abs_path(directory_path)

    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    if len(files) == 1 and os.path.isdir(os.path.join(path, files[0])):
        subdir_path = os.path.join(path, files[0])
        # copy all the files in the subdirectory to the parent one
        for file_name in os.listdir(subdir_path):
            shutil.move(src=os.path.join(subdir_path, file_name), dst=path)
        # done forget to delete the subdirectory
        os.rmdir(subdir_path)


def copy_directories(src_dir: str, des_dir: str, copy: bool = True,
                     filter_directories: callable = None) -> None:
    # convert the src_dir and des_dir to absolute paths
    src_dir, des_dir = abs_path(src_dir), abs_path(des_dir)

    assert os.path.isdir(src_dir) and os.path.isdir(des_dir), "BOTH ARGUMENTS MUST BE DIRECTORIES"

    if filter_directories is None:
        def filter_directories(x):
            return True

    # iterate through each file in the src_dir
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)
        if filter_directories(file_name):
            if copy:
                shutil.copy(file_path, des_dir)
            else:
                shutil.move(file_path, des_dir)

    # remove the source directory if it is currently empty
    if os.listdir(src_dir) == 0:
        os.rmdir(src_dir)


def unzip_data_file(data_zip_path: Union[Path, str], remove_inner_zip_files: bool = True) -> Path:
    data_zip_path = abs_path(data_zip_path)

    assert os.path.exists(data_zip_path), "MAKE SURE THE DATA'S PATH IS SET CORRECTLY!!"

    current_dir = Path(data_zip_path).parent
    unzipped_dir = os.path.join(current_dir, os.path.basename(os.path.splitext(data_zip_path)[0]))
    os.makedirs(unzipped_dir, exist_ok=True)

    # let's first unzip the file
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        # extract the data to the unzipped_dir
        zip_ref.extractall(unzipped_dir)

    # unzip any files inside the subdirectory
    for file_name in os.listdir(unzipped_dir):
        file_path = os.path.join(unzipped_dir, file_name)
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # extract the data to current directory
                zip_ref.extractall(unzipped_dir)

        # remove the zip files if the flag is set to True
        if remove_inner_zip_files:
            os.remove(file_path)

    # squeeze all the directories
    for file_name in os.listdir(unzipped_dir):
        squeeze_directory(os.path.join(unzipped_dir, file_name))

    return unzipped_dir
