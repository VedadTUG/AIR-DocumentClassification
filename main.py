from preProcessing import PreProcessor
from naiveBayesClassifier import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from rnn import create_dataset
from rnn import RNN
from predictions import MakePredictions
from training import TrainModel
from cnn import CNN




def calculate_tf_idf(documents: list[str], query: str, number_of_documents : int):
    tfidf_vectoriser = TfidfVectorizer()
    tf_idf_matrix = tfidf_vectoriser.fit_transform(documents)
    query_vector = tfidf_vectoriser.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tf_idf_matrix).flatten()
    cosine_similarities_sorted = cosine_similarities.argsort()[::-1]
    for counter, entry in enumerate(cosine_similarities_sorted):
        if counter == number_of_documents:
            break
        print(f"Similarity: {cosine_similarities[entry]}, Document: {documents[entry]}")
    return



preprocessor = PreProcessor()
data_train, data_test = preprocessor.make_sets()
complete_set_data = data_train + data_test
lookup_table = preprocessor.create_lookup_table(complete_set_data)
#print(lookup_table)
data_train_indexed = preprocessor.convert_set_into_word_indexes(data_train, lookup_table, 0)
length = preprocessor.get_longest_list(data_train_indexed) #13820
data_train_single_list = preprocessor.preprocess_tokenise_single_entry(data_train[0])




max_length = 50
embedding_dim = 100
hidden_dim = 128
num_classes = 20
batch_size = 32
vocab_size = len(lookup_table) + 1
epochs = 15
learning_rate = 0.001
k = 5

#For CNN only
filter_sizes=[10,20,30]
num_filters=100
dropout=0.5

loss_fn = nn.CrossEntropyLoss()
rnn_classifier = RNN(vocab_size, embedding_dim, hidden_dim, num_classes)
optimizer_rnn = torch.optim.Adam(rnn_classifier.parameters(), lr=learning_rate)
cnn_classifier = CNN(vocab_size, embedding_dim, num_classes, dropout, num_filters, filter_sizes)
optimizer_cnn = torch.optim.Adam(cnn_classifier.parameters(), lr=learning_rate)


queries_text = preprocessor.extract_strings_from_csv('query_text', 'data/queries.csv')
queries_labels = preprocessor.extract_strings_from_csv('query_label', 'data/queries.csv')
queries_text = preprocessor.preprocess_queries(queries_text)

query_data_labeled = preprocessor.create_query_set(queries_text, queries_labels)
converted_queries = preprocessor.convert_set_into_word_indexes_query(query_data_labeled, lookup_table, max_length)
#calculate_tf_idf(complete_set_data, queries_text[0], 5)
train_data_labeled, test_data_labeled = preprocessor.make_labels()
converted_doc_train = preprocessor.convert_set_into_word_indexes(train_data_labeled, lookup_table, max_length)
converted_doc_test = preprocessor.convert_set_into_word_indexes(test_data_labeled, lookup_table, max_length)


train_dataset = create_dataset(converted_doc_train)
test_dataset = create_dataset(converted_doc_test)
val_dataset = create_dataset(converted_queries)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


naivebayes = NaiveBayesClassifier()
naivebayes.naiveBayes(train_data_labeled, test_data_labeled, k)

TrainModel(rnn_classifier, loss_fn, optimizer_rnn, train_loader, test_loader, epochs)
MakePredictions(rnn_classifier, val_loader, k)

TrainModel(cnn_classifier, loss_fn, optimizer_cnn, train_loader, test_loader, epochs)
MakePredictions(cnn_classifier, val_loader, k)