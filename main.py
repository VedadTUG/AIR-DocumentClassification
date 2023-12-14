import numpy as np
import pandas as pd
import tqdm as tqdm
import torch as torch
import sklearn
import nltk
import matplotlib as plt
import seaborn as sns
import os
from pathlib import Path

def read_data(path):
    list_of_entries = []
    for filepath in Path(path).glob('*'):
        list_of_entries.append(open(filepath, encoding="utf-8", errors='ignore').read())
    return list_of_entries

path = 'data/alt.atheism'
print(len(read_data(path)))