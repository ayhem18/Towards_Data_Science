"""This script contains functionalities for building ready to use vocabularies using the torchtext library from the
Pytorch framework"""

# let's build our vocabulary
# first we can use a good tokenizer from nltk
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
import torchtext
import re
from typing import Union


def preprocess_sentence(s: str) -> list[str]:
    s = s.strip().lower()
    # remove unnecessary characters
    s = re.sub(r"[^a-zA-Z\s.,!?()'\"]+", "", s)
    # remove double spaces
    return re.sub(r'\s+', ' ', s).strip()


def default_tokenizer(sentence: str) -> list[str]:
    return word_tokenize(preprocess_sentence(sentence))


# check the documentation here:
# https://pytorch.org/text/stable/vocab.html#build-vocab-from-iterator

def vocab_iterator(text_iterable) -> list[str]:
    for text in text_iterable:
        yield word_tokenize(preprocess_sentence(text))


def build_vocabulary(train_data, max_size: int = 10000,
                     special_tokens: Union[list[str], str] = None) -> torchtext.vocab:
    # set the default value for the special tokens
    if special_tokens is None:
        special_tokens = ["<unk>"]
    # convert str to a list of only one element if needed
    special_tokens = [special_tokens] if isinstance(special_tokens, str) else special_tokens
    vocab = build_vocab_from_iterator(iterator=vocab_iterator(train_data), max_tokens=max_size, specials=special_tokens,
                                      special_first=True)
    # let's set the default index to the first special token
    vocab.set_default_index(vocab[special_tokens[0]])
    return vocab
