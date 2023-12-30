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


target_classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
         'comp.sys.mac.hardware',
         'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
         'rec.sport.hockey',
         'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
         'talk.politics.guns',
         'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

test_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


#https://chat.openai.com/share/62bbba47-733d-406b-be41-588959e64f77
def calculateF1K(Y_pred, Y_true, k):
    _, indices = torch.topk(Y_pred, k, dim=1)
    top_k_mask = torch.zeros_like(Y_pred)
    top_k_mask.scatter_(1, indices, 1)

    batch_size = Y_true.size(0)
    # Create a one-hot tensor from y_true
    y_true_one_hot = torch.zeros(batch_size, 20)
    y_true_one_hot[torch.arange(batch_size), Y_true] = 1

    # Create a one-hot tensor from y_pred_indices
    y_pred = torch.zeros(batch_size, 20)
    row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, k)
    y_pred[row_indices, indices] = 1

    true_positives = (y_true_one_hot * y_pred).sum(dim=1)
    predicted_positives = y_pred.sum(dim=1)
    actual_positives = y_true_one_hot.sum(dim=1)

    precision = true_positives / (predicted_positives + 1e-10)  # Adding epsilon to avoid division by zero
    recall = true_positives / (actual_positives + 1e-10)  # Adding epsilon to avoid division by zero

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)  # Adding epsilon to avoid division by zero

    return f1_score.mean().item()


#https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-rnn-for-text-classification-tasks
def MakePredictions(model, loader, kmost = 0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        token_tensor = torch.stack(X)
        token_tensor = token_tensor.permute(1, 0)
        token_tensor = token_tensor.to(device)
        target_tensor = Y
        target_tensor = target_tensor.to(device)
        preds = model(token_tensor.round())
        Y_preds.append(preds)
        Y_shuffled.append(target_tensor)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds).cpu(), torch.cat(Y_shuffled).cpu()


    f1k= calculateF1K(Y_preds, Y_shuffled, kmost)



    Y_actual, Y_preds =  Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()


    print(f"General F1@{kmost} is:")
    print(f1k)
    print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
    print("\nClassification Report : ")
    print(classification_report(Y_actual, Y_preds, target_names=target_classes,  zero_division='warn'))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_actual, Y_preds))

    skplt.metrics.plot_confusion_matrix([test_classes[i] for i in Y_actual], [test_classes[i] for i in Y_preds],
                                        normalize=True,
                                        title="Confusion Matrix",
                                        cmap="Purples",
                                        hide_zeros=True,
                                        figsize=(15, 15)
                                        )

    plt.show()
