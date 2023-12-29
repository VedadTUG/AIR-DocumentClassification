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

embed_len = 150
hidden_dim = 50
n_layers=1
n_filters = 3
filter_sizes=[1, 5, 10, 15]
num_filters=10
num_classes=20
dropout=0.2
vocab_size = 130317


#https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
#new class: https://www.kaggle.com/code/mlwhiz/multiclass-text-classification-pytorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO: Add your layers below
        self.embedding = nn.Embedding(vocab_size, embed_len)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_len)) for K in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        #scaled_output = 19 * torch.sigmoid(logit)
        return logit



    def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs):


        for i in range(1, epochs+1):
            model.train()
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
                    target_tensor = Y
                    preds = model(token_tensor)
                    loss = loss_fn(preds, target_tensor)
                    losses.append(loss.item())

                    Y_shuffled.append(target_tensor)
                    Y_preds.append(preds.argmax(dim=-1))

                Y_shuffled = torch.cat(Y_shuffled)
                Y_preds = torch.cat(Y_preds)

                print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
                print(
                    "Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))