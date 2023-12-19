import numpy as np
import pandas as pd
import torch as torch
import sklearn
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

def read_data(path):
    final_list_of_blogs = []

    for filepath in Path(path).glob('*'):
        with open(filepath, 'r', encoding="utf-8", errors='ignore') as blog_entry:
            blog_line = blog_entry.readlines()
        final_list_of_blogs.append(blog_line)
    return final_list_of_blogs

def create_string_from_data(data: list[str]):
    final_list = []
    for blog_entry in data:
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

def preprocess_lowercase(data: list[str]):
    final_list = []
    for blog_entry in data:
        final_list.append(blog_entry.lower())
    return final_list

def preprocess_tokenize(data: list[str]):
    final_list = []
    for blog_entry in data:
        final_list.append(word_tokenize(blog_entry))
    return final_list

def preprocess_tokenise_single_entry(entry: str):
    return word_tokenize(entry)

def preprocess_remove_stopwords(data: list[list[str]]):
    final_list = []
    for blog_entry in data:
        stop_words = stopwords.words('english')
        filtered_words = [word for word in blog_entry if word not in stop_words]
        filtered_text = ' '.join(filtered_words)
        final_list.append(filtered_text)
    return final_list

def preprocess_remove_numbers(data: list[str]):
    final_list = []
    for blog_entry in data:
        regex_numbers = '[0-9]'
        entry_without_numbers = re.sub(regex_numbers, '', blog_entry)
        final_list.append(entry_without_numbers)
    return final_list

def preprocess_remove_punctuation(data: list[str]):
    final_list = []
    for blog_entry in data:
        regex_punctuation = '[^\w\s]'
        entry_without_punctuation = re.sub(regex_punctuation, '', blog_entry)
        final_list.append(entry_without_punctuation)
    return final_list

def preprocess_stem_words(data: list[str]):
    final_list = []
    porter_stemmer = PorterStemmer()

    for blog_entry in data:
        tokenised_entry = preprocess_tokenise_single_entry(blog_entry)
        stemmed_entry = [porter_stemmer.stem(word) for word in tokenised_entry]
        stemmed_string = ' '.join(stemmed_entry)
        final_list.append(stemmed_string)
    return final_list


def preprocess_data(path: str):
    data = read_data(path)
    data = create_string_from_data(data)
    data = remove_headers(data)
    data = preprocess_lowercase(data)
    data = preprocess_tokenize(data)
    data = preprocess_remove_stopwords(data)
    data = preprocess_remove_numbers(data)
    data = preprocess_remove_punctuation(data)
    data = preprocess_stem_words(data)

    return data

def create_train_and_test_set():
    root = 'data/'
    train_extension = '20news-bydate-train/'
    test_extension = '20news-bydate-test/'
    paths = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
             'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
             'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
             'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    print("Preprocessing train data")
    train_set = []
    for path in tqdm(paths):
        final_path = root+train_extension+path
        path_data = preprocess_data(final_path)
        for blog_post in path_data:
            train_set.append(blog_post)

    print("Preprocessing test data")
    test_set = []
    for path in tqdm(paths):
        final_path = root + test_extension + path
        path_data = preprocess_data(final_path)
        for blog_post in path_data:
            test_set.append(blog_post)



    return train_set, test_set

def make_sets():
    if os.path.exists('data/data_train.pkl') and os.path.exists('data/data_test.pkl'):
        print('Pickle file already exists, loading data!')
        with open('data/data_train.pkl', 'rb') as file1, open('data/data_test.pkl', 'rb') as file2:
            data_train = pickle.load(file1)
            data_test = pickle.load(file2)
    else:
        print('Pickle file does not exist, creating one for easier loading!')
        data_train, data_test = create_train_and_test_set()
        with open('data/data_train.pkl', 'wb') as file1, open('data/data_test.pkl', 'wb') as file2:
            pickle.dump(data_train, file1)
            pickle.dump(data_test, file2)
    return data_train, data_test

labels = [
    'alt_atheism', 'comp_graphics', 'comp_ms_windows_misc', 'comp_ibm', 'comp_mac', 'comp_windows', 'misc_forsale',
    'rec_autos', 'rec_motorcycles', 'rec_sport_basketball', 'roc_sport_hockey', 'sci_crypt', 'sci_electronics',
    'sci_med', 'sci_space', 'soc_religion_christian', 'politics_guns', 'politics_mideast', 'politics_misc,', 'religion_misc'
        ]

data_train, data_test = make_sets()







