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


labels = [
    'alt_atheism', 'comp_graphics', 'comp_ms_windows_misc', 'comp_ibm', 'comp_mac', 'comp_windows', 'misc_forsale',
    'rec_autos', 'rec_motorcycles', 'rec_sport_basketball', 'roc_sport_hockey', 'sci_crypt', 'sci_electronics',
    'sci_med', 'sci_space', 'soc_religion_christian', 'politics_guns', 'politics_mideast', 'politics_misc,', 'religion_misc'
        ]



