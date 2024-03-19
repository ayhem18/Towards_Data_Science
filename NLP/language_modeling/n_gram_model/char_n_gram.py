"""
"""
import os, re

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Sequence, Union, Iterator
from collections import defaultdict, Counter
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# START_CHAR = "S"
# END_CHAR = "E"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class NGramModel:

    def _build_n_gram_model(self, data: Union[Sequence[str], Iterator[str]]):
        map = defaultdict(lambda :Counter())
        
        for word in tqdm(data, desc='processing semantic units'):
            chars = self.start_char + word + self.end_char
            # the main idea here
            for i in range(len(chars) - self.n + 1):
                # update the count of the n-th character by one
                map[chars[i: i + self.n - 1]].update(chars[i + self.n - 1])

            self.vocab.update(chars)
        
        self.vocab.remove(self.start_char)
        self.vocab.remove(self.end_char)
        self.vocab = [self.start_char, self.end_char] + sorted(self.vocab) 
        self.char2i = {c: i for (i, c) in enumerate(self.vocab)}
        self.index2c = {i: c for (i, c) in enumerate(self.vocab)}

        # now we are ready to proceed with the compuations
        for key, value in map.items(): 
            # iterate through the vocabulary
            for c in self.vocab:
                if c not in value: 
                    value[c] = 0 
            map[key] = [v for _, v in sorted(value.items(), key=lambda x: x[0])] 
        
        self.count_model = pd.DataFrame(data=map, index=self.vocab).T
        self.prob_model = self.count_model / self.count_model.sum(axis=1)

        # few checks: 
        all_positive = np.sum(self.prob_model.values >= 0) == self.prob_model.size
        if not all_positive: 
            raise ValueError(f"Please make sure all the numbers are positive as they should represent probabilities")

        sum_1_by_row = np.allclose(self.prob_model.values.sum(axis=1), np.ones(len(self.prob_model),), atol=10 ** -6)
        if not sum_1_by_row:
            raise ValueError(f"Please make sure the values sum up to 1 for each row")

    
    def _build_model_from_file(self, file_path: Union[str, Path]):
        if not os.path.exists(file_path): 
            raise ValueError(f"Please make sure to pass a correct path to an exisiting file to this method\n{file_path} does not exist")

        def _file_iterator(file):
            with open(file_path, 'r') as f:
                for f in f.readlines():
                    # some simple preprocessing: 
                    # lowering, remove multiple spaces
                    # remove training and leading spaces
                    f = re.sub('\s+', ' ', f.strip().lower())
                    if len(f) > 0:
                        # split the text into sentences
                        sentences = sent_tokenize(f)
                        for s in sentences: 
                            yield s 
        # build the n-gram using the iterator 
        self._build_n_gram_model(data=_file_iterator)

    def __init__(self, 
                 data: Union[str, Path, Sequence[str]],
                 n: int,
                 vocab: set[str] = None,
                 start_char: str = 'S',
                 end_char: str = 'E', 
                 local_copy: bool = True
                 ) -> None: 
        # the training data
        self.data = data
        # the n-gram
        self.n = n

        # the vocabulary used by the model
        self.vocab = vocab

        # special characters used to denote the start and end of a piece of text
        self.start_char = start_char
        self.end_char = end_char

        # we consistently need to map between characters and their indices
        self.char2i, self.index2c= {}, {}
        
        # we will have 2 versions: count model and prob model
        self.count_model: pd.DataFrame = None
        self.prob_model: pd.DataFrame = None

        # now we are ready to proceed with building the n-gram model
        if isinstance(data, (str, Path)):
            self._build_model_from_file(data)
        else: 
            self._build_n_gram_model(data=data)

        # make sure to save the dataframe locally for convenience
        if local_copy:
            self.count_model.to_csv(os.path.join(SCRIPT_DIR, f'{n}_gram_model.csv'), index=True)

    def _sample(self, max_length: int):
        """This function generates a random sequence of strings using the n-gram model. The generation stops either after generating the END character
        or hitting the maximum number of characters in the sequence. The generation is carried out as follows: 
        
        1. generate the first n - 1 characters that start with the self.start_char 
        2. given the last n-1 consider the conditional distribution P(char | (last n - 1 chars))
        3. use a random multinomial random generator to sample the next char
        
        Args:
            max_length the maximum number of charactes in the sequence !!
        """

        if max_length < self.n - 1:
            raise NotImplementedError(f"The current implementation supports generating sequences of length at least {self.n - 1}")

        # step 1: generate the first n - 1 chars
        
        # extract the n - 1 sequence of chars that start with the start characters, and calculate the overall occurrence
        start_sequences = self.count_model.loc[self.count_model.index.str.startswith(self.start_char), :]
        # convert them to probabilities
        n_grams_start_prob = (start_sequences.sum(axis=1) / start_sequences.sum()).values

        # sample the first n - 1 chars
        generator = np.random.Generator()
        result = start_sequences.index[generator.multinomial(n=1, pvals=n_grams_start_prob).item()]

        # now we proceed with generating the rest of the sequence: 
        while len(result) < max_length:
            last_chars = result[-(self.n - 1):]
            # use the last  n - 1 to determine the next 
            probs = self.prob_model.loc[last_chars, :].values
            # sample
            next_char = self.index2c[generator.multinomial(n=1, pvals=probs).item()]

            # if the sampled character is the end character, the job here is done !!
            if next_char == self.end_char: 
                break
            result += next_char

        return result
    
# def build_n_gram_language_model(words_bank: Union[Sequence[str], Iterator[str]], 
#                                 n: int = 2, 
#                                 vocab: set = None) -> pd.DataFrame: 
#     """
#     This function builds a n-gram model of the given dataset by computing the probability P(nk | n1&n2&n3...&nk) = Count(n1,n2, ..nk) / Count(n1, n2, ... n{k - 1})
#     """    
#     if n >= 5:
#         raise ValueError(f"The bi-gram model does not work well with 'n' larger than 5.")
    
#     if vocab is None:
#         vocab = set()

#     map = defaultdict(lambda :Counter())
    
#     for word in tqdm(words_bank, desc='processing semantic units'):
#         chars = START_CHAR + word + END_CHAR
#         # the main idea here
#         for i in range(len(chars) - n + 1):
#             # update the count of the n-th character by one
#             map[chars[i: i + n - 1]].update(chars[i + n - 1])

#         vocab.update(chars)

#     vocab = sorted(list(vocab))

#     for key, value in map.items(): 
#         # iterate through the vocabulary
#         for c in vocab:
#             if c not in value: 
#                 value[c] = 0 
#         map[key] = [v for _, v in sorted(value.items(), key=lambda x: x[0])] 
    
#     data = pd.DataFrame(data=map, index=vocab).T
#     return data, vocab

# def sample(n_gram_model: pd.DataFrame, 
#            n: int,
#            vocab: Sequence[str],
#            max_length: int = None, 
#            seed: int = 69) -> str:
#     np.random.seed(seed=seed)
#     assert max_length >= n - 1, "The maximum length should be at least equal to n - 1 in n-gram models"
#     # the main idea is to randomly sample a sequence of characters using the n_gram_model
#     # the current implementation supports generating only an output with of length at least 'n'

#     normalized_n_gram = n_gram_model / n_gram_model.sum(axis=1)

#     # few checks: 
#     all_positive = np.sum(normalized_n_gram.values >= 0) == normalized_n_gram.size
#     if not all_positive: 
#         raise ValueError(f"Please make sure all the numbers are positive as they should represent probabilities")
    
#     sum_1_by_row = np.allclose(normalized_n_gram.values.sum(axis=1), np.ones(len(normalized_n_gram),), atol=10 ** -6)
#     if not sum_1_by_row:
#         raise ValueError(f"Please make sure the values sum up to 1 for each row")

#     # let's start by sampling the first n - 1 characters
#     n_grams_start_count = n_gram_model.loc[n_gram_model.index.str.startswith(START_CHAR), :].sum(axis=1)
#     n_grams_start_probs = n_gram_model.loc[n_gram_model.index.str.startswith(START_CHAR), :] / n_grams_start_count

#     # sample a start at random
#     result = n_grams_start_probs.index[np.random.Generator.multinomial(1, n_grams_start_probs.values)]
#     char = ""
#     while len(result) <= max_length and char != END_CHAR:
#         probs = n_gram_model.loc[result[-n:], :]
#         result += vocab[np.random.Generator.multinomial(1, probs)]

#     result

if __name__ == '__main__':
    pass
