from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

class NaiveBayesClassifier:
    def __init__(self):
        pass

    def naiveBayes(self, train_set: list[str, list[str]], test_set: list[str, list[str]], k=None):
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
        predicted_probabilities = naive_bayes_classifier.predict_proba(tf_idf_matrix_test)

        #https://chat.openai.com/share/d6fd0206-d4c1-4e9c-9817-c743b1f26751
        if k is not None:
            top_k_predictions = []
            for probs in predicted_probabilities:
                top_k_indices = probs.argsort()[-k:][::-1]  # Indices of top-k predictions
                top_k_predictions.append(naive_bayes_classifier.classes_[top_k_indices])

            # Calculate F1@k
            true_labels = [[label] for label in test_labels]
            top_k_predictions = [list(pred) for pred in top_k_predictions]
            f1_at_k = 0.0
            for true, pred in zip(true_labels, top_k_predictions):
                intersection = len(set(true) & set(pred))
                precision = intersection / k if k > 0 else 0
                recall = intersection / len(true)
                f1_at_k += 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0

            f1_at_k /= len(true_labels)  # Average F1@k over all samples
            print("F1@k for Test Set (Top-{}):".format(k))
            print(f1_at_k)


        print("Classification Report on Test Set:")
        print(classification_report(test_labels, predicted_labels))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(test_labels, predicted_labels))
