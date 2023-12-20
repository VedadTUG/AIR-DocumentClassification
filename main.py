from preProcessing import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


preprocessor = PreProcessor()
data_train, data_test = preprocessor.make_sets()
complete_set_data = data_train + data_test
lookup_table = preprocessor.create_lookup_table(complete_set_data)
data_train_indexed = preprocessor.convert_set_into_word_indexes(data_train, lookup_table)
queries_text = preprocessor.extract_strings_from_csv('query_text', 'data/queries.csv')
queries_text = preprocessor.preprocess_queries(queries_text)
calculate_tf_idf(complete_set_data, queries_text[0], 5)















