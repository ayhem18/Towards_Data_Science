"""
_summary_
"""
import os, requests

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, 'data')

_small_shakespeare_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def download_small_check():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(os.path.join(DATA_FOLDER, 'small_shake.txt'), 'w') as f:
        f.write(requests.get(_small_shakespeare_url).text)

def prepare_data_small_check(train_split: float = 0.9, 
                             write_every_n_lines: int = 10000):
    # to make this function general, we will iterate through the file twice, determine the total number of lines
    # and then split the data to train and test splits
    total_lines_count = 0
    with open(os.path.join(DATA_FOLDER, 'small_shake.txt'), 'r') as f:
        for f in f.readlines():
            total_lines_count += int(len(f.strip()) != 0)

    train_line_count = int(train_split * total_lines_count)

    # read the file again
    with open(os.path.join(DATA_FOLDER, 'small_shake.txt'), 'r') as fr:
        i = 0 
        train_data, val_data = "", ""
        with open(os.path.join(DATA_FOLDER, 'small_shake_train.txt'), 'a') as ft:   
            with open(os.path.join(DATA_FOLDER, 'small_shake_val.txt'), 'a') as fv:
                for l in fr.readlines():
                    i +=  int(len(l.strip()) >= 0)
                    if i <= train_line_count:
                        train_data += l.strip() + "\n"
                    else: 
                        val_data += l.strip() + "\n"

                # save both
                ft.write(train_data)
                fv.write(val_data)


if __name__ == '__main__':
    download_small_check()
    prepare_data_small_check()
