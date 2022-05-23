"""
Preprocess the data to be trained by the learning algorithm.
Create files `preprocessor.joblib` and `preprocessed_data.joblib`
"""

import os
import pandas as pd
import numpy as np
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from joblib import dump, load

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text


def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r'(\S+)')

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer


def main():
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    prepared_questions = []
    with open('data/text_prepare_tests.tsv', encoding='utf-8') as prepared_tests_file:
        for line in prepared_tests_file.readlines():
            line = text_prepare(line.strip())
            prepared_questions.append(line)
    text_prepare_results = '\n'.join(prepared_questions)

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    X_train_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_features(X_train, X_val, X_test)

    # @todo Make use of preprecessor(pipeline of tfidf) of sklearn

    if not os.path.exists('output'):
        os.makedirs(os.getcwd() + '/output', exist_ok=True)

    dump(X_train_mybag, 'output/preprocessed_x_train_mybag.joblib')
    dump(X_val_mybag, 'output/preprocessed_x_val_mybag.joblib')
    dump(X_test_mybag, 'output/preprocessed_x_test_mybag.joblib')

    dump(X_train_tfidf, 'output/preprocessed_x_train_tfidf.joblib')
    dump(X_val_tfidf, 'output/preprocessed_x_val_tfidf.joblib')
    dump(X_test_tfidf, 'output/preprocessed_x_test_tfidf.joblib')
    dump(tfidf_vectorizer.vocabulary_, 'output/tfidf_vocab.joblib')
    dump(tfidf_vectorizer, 'output/tfidf_vectorizer.joblib')

    dump(X_train, 'output/X_train.joblib')
    dump(X_val, 'output/X_val.joblib')
    dump(y_train, 'output/y_train.joblib')
    dump(y_val, 'output/y_val.joblib')
    dump(tags_counts, 'output/tags_counts.joblib')
    dump(WORDS_TO_INDEX, 'output/WORDS_TO_INDEX.joblib')


if __name__ == "__main__":
    main()
