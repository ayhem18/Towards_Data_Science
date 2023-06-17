"""This script contains functionalities to build and a train a simply toy Feed Forward Neural Network as Language Model: FFNLM
"""
import torch
from torch import nn
from collections import OrderedDict
from _collections_abc import Sequence
import torchtext
from building_vocabulary import default_tokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from math import ceil
from torchtext.datasets import WikiText2
from building_vocabulary import build_vocabulary


DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'


class NeuralLM(nn.Module):
    min_layer_units = 16

    @classmethod
    def output_size(cls, input_size: int) -> int:
        return max(cls.min_layer_units, int(input_size / 2))

    def __init__(self, vocab_size: int, dims_embedding: int = 32, num_layers: int = 3,
                 window_size: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert num_layers >= 1
        self.vocab_size = vocab_size
        self.dims_embedding = dims_embedding
        self.window_size = window_size

        # the first step is to create an embedding layer
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dims_embedding)
        # create the block of stacked layers
        # the first linear layer should have windows_size * dims_embeddings as input size
        input_size = self.window_size * self.dims_embedding
        linear_block_dict = OrderedDict()

        for i in range(1, num_layers):
            linear_block_dict[f'layer_{i}'] = nn.Linear(in_features=input_size,
                                                        out_features=self.output_size(input_size))
            linear_block_dict[f'relu_{i}'] = nn.ReLU()
            input_size = self.output_size(input_size)

        # the last layer is not followed by a RELU activation function, and has a different number output units: 
        # num of classes: the size of the vocabulary. Hence, it should be proceeded separately      
    
        linear_block_dict[f'layer_{num_layers}'] = nn.Linear(in_features=input_size,
                                                             out_features=self.vocab_size)  

        self.nn_block = nn.Sequential(linear_block_dict)

    # time to define the forward function
    def forward(self, x: torch.tensor):
        # first convert the input to embeddings and change the shape in a way that concatenates the 'self.window_size' embeddings together
        x_embedded = self.embeddings(x).view(-1, self.window_size * self.dims_embedding)
        # pass it through the nn.block
        return self.nn_block(x_embedded)


# let's get the data prepared for our model
def collate_function(batch_text: Sequence[str], vocabulary: torchtext.vocab,
                     window_size: int, tokenizer: callable = None) -> tuple[torch.LongTensor, torch.LongTensor]:
    # the default tokenizer will be the one used to build the vocabulary
    if tokenizer is None:
        tokenizer = default_tokenizer

    # input and output as lists
    input_list, output_list = [], []
    # iterate through the sentences in the batch
    for sentence in batch_text:
        # convert to tokens
        tokens = tokenizer(sentence)
        # convert the tokens to numerical indices
        tokens_ids = vocabulary.lookup_indices(tokens)
        # use the window_size to extract the data needed:
        input_list.extend([tokens_ids[i: i + window_size] for i in range(len(tokens) - window_size)])
        # add the labels
        output_list.extend([tokens_ids[i] for i in range(window_size, len(tokens))])
        
        # make sure the input matches the output
        assert len(output_list) == len(input_list)

    # convert each of the lists to pytorch tensors
    input_tensor = torch.tensor(input_list, dtype=torch.long)
    output_tensor = torch.tensor(output_list, dtype=torch.long)

    return input_tensor, output_tensor


def train_NLM(model: NeuralLM, vocab: torchtext.vocab, train_data, train_data_size: int, epochs: int, batch_size: int,
              device: str = DEVICE, tokenizer: callable = None) -> None:

    # before setting the data loader, we need to set the collate-function as it is parametrized
    def custom_collate_fn(batch_text: str):
        return collate_function(batch_text, vocabulary=vocab, window_size=model.window_size,
                                tokenizer=tokenizer)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  collate_fn=custom_collate_fn, shuffle=True, drop_last=True)

    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    for epoch in range(epochs):
        for _, batch_data in enumerate(tqdm(train_dataloader, total = int(ceil(train_data_size / batch_size)))):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch + 1}/{epochs}, Train Loss={loss.item()}")


# let's write the driver's code
def driver_code():
    train_data, val_data, test_data = WikiText2(root='data') # this will create a directory called data where all the 'data' will be stored
    DATASET_SIZE = 36718 # check the documentation here: https://pytorch.org/text/stable/datasets.html#language-modeling
    vocab = build_vocabulary(train_data)
    language_model = NeuralLM(len(vocab))
    train_NLM(language_model, vocab=vocab, train_data=train_data, train_data_size=DATASET_SIZE, epochs=20, batch_size=8)


if __name__ == '__main__':
    driver_code()
    