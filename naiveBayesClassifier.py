from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class NaiveBayesClassifier:
    def __init__(self):
        pass

    def naiveBayes(self, train_set: list[str, list[str]], test_set: list[str, list[str]]):
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
