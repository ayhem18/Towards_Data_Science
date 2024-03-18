"""
"""

import numpy as np
import pandas as pd

from typing import Sequence
from collections import defaultdict, Counter
from tqdm import tqdm

START_CHAR = "S"
END_CHAR = "E"

def build_n_gram_language_model(words_bank: Sequence[str], 
                                n: int = 2, 
                                vocab: set = None) -> pd.DataFrame: 
    """
    This function builds a n-gram model of the given dataset by computing the probability P(nk | n1&n2&n3...&nk) = Count(n1,n2, ..nk) / Count(n1, n2, ... n{k - 1})
    """    
    if n >= 5:
        raise ValueError(f"The bi-gram model does not work well with 'n' larger than 5.")
    
    if vocab is None:
        vocab = set()

    map = defaultdict(lambda :Counter())
    
    for word in tqdm(words_bank, desc='processing semantic units'):
        chars = START_CHAR + word + END_CHAR
        # the main idea here
        for i in range(len(chars) - n + 1):
            # update the count of the n-th character by one
            map[chars[i: i + n - 1]].update(chars[i + n - 1])

        vocab.update(chars)
    
    vocab = sorted(list(vocab))

    for key, value in map.items(): 
        # iterate through the vocabulary
        for c in vocab:
            if c not in value: 
                value[c] = 0 
        map[key] = [v for _, v in sorted(value.items(), key=lambda x: x[0])] 
    
    data = pd.DataFrame(data=map, index=vocab).T
    return data


if __name__ == '__main__':
    words_bank = ['abcd', 'aaec', 'aaeb']
    vocab = {'a', 'b', 'c', 'e', 'd'}
    