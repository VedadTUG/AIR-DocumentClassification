from preProcessing import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np

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

def naiveBayes(train_set: list[str, list[str]], test_set: list[str, list[str]]):
    train_labels = []
    train_entries = []
    for label, entry in train_set:
        train_labels.append(label)
        train_entries.append(entry)

    test_labels = []
    test_entries = []
    for label, entry in test_set:
        test_labels.append(label)
        test_entries.append(entry)

    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_matrix_train = tf_idf_vectorizer.fit_transform(train_entries)
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(tf_idf_matrix_train, train_labels)

    tf_idf_matrix_test = tf_idf_vectorizer.transform(test_entries)

    predicted_labels = naive_bayes_classifier.predict(tf_idf_matrix_test)

    print("Classification Report on Test Set:")
    print(classification_report(test_labels, predicted_labels))



preprocessor = PreProcessor()
data_train, data_test = preprocessor.make_sets()
complete_set_data = data_train + data_test
lookup_table = preprocessor.create_lookup_table(complete_set_data)
data_train_indexed = preprocessor.convert_set_into_word_indexes(data_train, lookup_table)
queries_text = preprocessor.extract_strings_from_csv('query_text', 'data/queries.csv')
queries_text = preprocessor.preprocess_queries(queries_text)
train_data_labeled, test_data_labeled = preprocessor.make_labels()

naiveBayes(train_data_labeled, test_data_labeled)













