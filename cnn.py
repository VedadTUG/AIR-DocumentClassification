import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch import optim
import torch.nn.functional as F
import nltk
import matplotlib as plt
import seaborn as sns
import os
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
from tqdm import tqdm
nltk.download('stopwords')
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset



embed_len = 50
hidden_dim = 50
n_layers=1
n_filters = 3
filter_sizes=[3, 4, 5]
num_filters=[100, 100, 100]
num_classes=20
dropout=0.5
#https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO: Add your layers below
        self.embedding = nn.Embedding(num_embeddings=130316,
                                      embedding_dim=embed_len,
                                      padding_idx=0,
                                      max_norm=5.0)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_len,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(x).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits


    def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs):

        model.train()
        for i in range(1, epochs+1):
            losses = []
            for batch, (X, Y) in enumerate(tqdm(train_loader)):
                token_tensor = torch.stack(X)
                token_tensor = token_tensor.permute(1, 0)
                token_tensor = token_tensor
                target_tensor = Y

                Y_preds = model(token_tensor)
                loss = loss_fn(Y_preds, target_tensor)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))

            model.eval()
            with torch.no_grad():
                Y_shuffled, Y_preds, losses = [], [], []
                for X, Y in val_loader:
                    token_tensor = torch.stack(X)
                    token_tensor = token_tensor.permute(1, 0)
                    token_tensor = token_tensor.to(torch.long)
                    target_tensor = Y
                    preds = model(token_tensor)
                    loss = loss_fn(preds, target_tensor)
                    losses.append(loss.item())

                    Y_shuffled.append(Y)
                    Y_preds.append(preds.argmax(dim=-1))

                Y_shuffled = torch.cat(Y_shuffled)
                Y_preds = torch.cat(Y_preds)

                print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
                print(
                    "Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))