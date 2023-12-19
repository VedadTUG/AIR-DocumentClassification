from preProcessing import PreProcessor


preprocessor = PreProcessor()
data_train, data_test = preprocessor.make_sets()
complete_set_data = data_train + data_test
lookup_table = preprocessor.create_lookup_table(complete_set_data)
data_train_indexed = preprocessor.convert_set_into_word_indexes(data_train, lookup_table)
print(data_train_indexed[0])
print(data_train[0])










