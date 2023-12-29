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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gc
import scikitplot as skplt


class create_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, target = self.data[index]
        return input, target



embed_len = 50
hidden_dim = 150
n_layers=3
vocab_size = 130317
output_classes = 20


#https://coderzcolumn.com/tutorials/artificial-intelligence/word-embeddings-for-pytorch-text-classification-networks
#https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-rnn-for-text-classification-tasks
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # TODO: Add your layers below
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_classes)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim))
        raw_output = self.linear(output[:,-1])
        #scaled_output = torch.sigmoid(raw_output)
        return raw_output



    def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs):


        for i in range(1, epochs+1):
            model.train()
            losses = []
            for batch, (X, Y) in enumerate(tqdm(train_loader)):
                token_tensor = torch.stack(X)
                token_tensor = token_tensor.permute(1, 0)
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


