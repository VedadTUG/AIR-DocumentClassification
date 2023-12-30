import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import nltk
import matplotlib as plt
import seaborn as sns
import os
import pickle
import re
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gc
import scikitplot as skplt

#embed_len = 150
#hidden_dim = 50
#n_layers=1
#n_filters = 3
filter_sizes=[3,4,5]
num_filters=10
#num_classes=20
dropout=0.5
#vocab_size = 130317


#https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
#new class: https://www.kaggle.com/code/mlwhiz/multiclass-text-classification-pytorch
#https://chat.openai.com/share/9a9d3ad2-b53f-43b4-92ef-846257d1b40f
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(CNN, self).__init__()
        # TODO: Add your layers below
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, embedding_dim)) for size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Apply embedding layer
        x = x.unsqueeze(1)  # Add 1 channel for convolution operation

        # Perform convolution and max pooling operations
        conv_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outputs = [torch.max(conv_output, 2)[0] for conv_output in conv_outputs]
        pooled_outputs = torch.cat(pooled_outputs, 1)

        x = self.dropout(pooled_outputs)
        x = self.fc(x)
        return x
