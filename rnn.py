import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import nltk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
from tqdm import tqdm
import torch
from torch import nn, optim

from torch.utils.data import DataLoader, Dataset



class create_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, target = self.data[index]
        return input, target



#https://coderzcolumn.com/tutorials/artificial-intelligence/word-embeddings-for-pytorch-text-classification-networks
#https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-rnn-for-text-classification-tasks
#https://medium.com/@spandey8312/text-classification-using-custom-data-and-pytorch-d88ba1087045
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNN, self).__init__()
        # TODO: Add your layers below
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


