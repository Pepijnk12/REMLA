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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, Pipeline
from joblib import dump, load

class TextModifier(BaseEstimator, TransformerMixin):
    """
    Custom text preperation transformer to fit in a pipeline
    """
    def __init__(self):
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.STOPWORDS = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.text_prepare(text) for text in X]

    def text_prepare(self, text):
        """
            text: a string

            return: modified initial string
        """
        text = text.lower()  # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = " ".join([word for word in text.split() if not word in self.STOPWORDS])  # delete stopwords from text
        return text

class MyBOWVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom BOW Transformer
    """
    def __init__(self, words_to_index=[], dict_size=5000):
        self.dict_size = dict_size
        self.words_to_index = words_to_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sp_sparse.vstack([sp_sparse.csr_matrix(self.my_bag_of_words(text, self.words_to_index, self.dict_size)) for text in X])

    def my_bag_of_words(self, text, words_to_index, dict_size):
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

def read_data(filename) -> pd.DataFrame:
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def get_words_count(X_train) -> dict:
    """
    Returns dictionaries of training data
    """
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1
    
    return words_counts

def get_tags_count(y_train) -> dict:
    """
    Returns dictionaries of training data
    """
    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    return tags_counts

def main():
    train = read_data('./data/external/train.tsv')
    validation = read_data('./data/external/validation.tsv')
    test = pd.read_csv('./data/external/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    
    words_counts = get_words_count(X_train)
    tags_counts = get_tags_count(y_train)

    # Create bow preprocessor pipeline
    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    ALL_WORDS = WORDS_TO_INDEX.keys()

    bow_preprocessor = Pipeline([
        ("text_modifier", TextModifier()),
        ("bow_vectorizer", MyBOWVectorizer(words_to_index=WORDS_TO_INDEX, dict_size=DICT_SIZE))
    ])

    # Create tfidf preprocessor pipeline
    tfidf_preprocessor = Pipeline([
        ("text_modifier", TextModifier()),
        ("tfidf_vectorizer", TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r'(\S+)'))
    ])

    X_train_mybow = bow_preprocessor.fit_transform(X_train)
    X_val_mybow = bow_preprocessor.transform(X_val)
    X_test_mybow = bow_preprocessor.transform(X_test)
    
    X_train_tfidf = tfidf_preprocessor.fit_transform(X_train)
    X_val_tfidf = tfidf_preprocessor.transform(X_val)
    X_test_tfidf = tfidf_preprocessor.transform(X_test)

    # Save preprocessors
    dump(bow_preprocessor, "./models/preprocessors/bow_preprocessor.joblib")
    dump(tfidf_preprocessor, "./models/preprocessors/tfidf_preprocessor.joblib")

    # Save preprocessed Data
    dump(X_train_mybow, './data/processed/preprocessed_x_train_mybow.joblib')
    dump(X_val_mybow, './data/processed/preprocessed_x_val_mybow.joblib')
    dump(X_test_mybow, './data/processed/preprocessed_x_test_mybow.joblib')

    dump(X_train_tfidf, './data/processed/preprocessed_x_train_tfidf.joblib')
    dump(X_val_tfidf, './data/processed/preprocessed_x_val_tfidf.joblib')
    dump(X_test_tfidf, './data/processed/preprocessed_x_test_tfidf.joblib')

    # Save raw data
    dump(X_train, './data/raw/X_train.joblib')
    dump(X_val, './data/raw/X_val.joblib')
    dump(y_train, './data/raw/y_train.joblib')
    dump(y_val, './data/raw/y_val.joblib')
    dump(tags_counts, './data/raw/tags_counts.joblib')
    dump(WORDS_TO_INDEX, './data/raw/WORDS_TO_INDEX.joblib')


if __name__ == "__main__":
    main()

