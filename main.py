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
    final_list_of_blogs = []

    for filepath in Path(path).glob('*'):
        with open(filepath, 'r', encoding="utf-8", errors='ignore') as blog_entry:
            blog_line = blog_entry.readlines()
        final_list_of_blogs.append(blog_line)
    return final_list_of_blogs

def create_string_from_data(data: list[str]):
    final_list = []
    for  blog_entry in data:
        string_for_entry = ' '
        for blog_line in blog_entry:
            string_for_entry = string_for_entry + blog_line
        final_list.append(string_for_entry)
    return final_list


def remove_headers(data: list[str]):
    final_blog_entries = []
    for blog_entry in data:

        index = blog_entry.find("Lines")
        if index != -1:
            lines_after_index = blog_entry[index:]
            lines_list = lines_after_index.splitlines()
            string_to_append = '\n'.join(lines_list[1:])
        final_blog_entries.append(string_to_append)
    return final_blog_entries

def preprocess_data(path: str):
    data = read_data(path)
    data_string = create_string_from_data(data)
    data_without_headers = remove_headers(data_string)
    return data_without_headers
path = 'data/20news-bydate-train/alt.atheism'

labels = [
    'alt_atheism', 'comp_graphics', 'comp_ms_windows_misc', 'comp_ibm', 'comp_mac', 'comp_windows', 'misc_forsale',
    'rec_autos', 'rec_motorcycles', 'rec_sport_basketball', 'roc_sport_hockey', 'sci_crypt', 'sci_electronics',
    'sci_med', 'sci_space', 'soc_religion_christian', 'politics_guns', 'politics_mideast', 'politics_misc,', 'religion_misc'
        ]
data = preprocess_data(path)
print(data[0])





