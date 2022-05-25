"""
Preprocess the data to be trained by the learning algorithm.
Create files `preprocessor.joblib` and `preprocessed_data.joblib`
"""

import os
import pandas as pd

from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from joblib import dump

from modules.text_preparer import TextPreparer
from modules.bow_vectorizer import BowVectorizer

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
    Returns dictionaries of label data
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

def create_bow_pipeline(words_to_index, dict_size) -> Pipeline:
    """
    Create bow preprocessor pipeline
    """
    bow_preprocessor = Pipeline([
        ("text_preparer", TextPreparer()),
        ("bow_vectorizer", BowVectorizer(words_to_index=words_to_index, dict_size=dict_size))
    ])

    return bow_preprocessor

def create_tfidf_pipeline() -> Pipeline:
    """
    Creates tfidf preprocessor pipeline
    """
    tfidf_preprocessor = Pipeline([
        ("text_preparer", TextPreparer()),
        ("tfidf_vectorizer", TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r'(\S+)'))
    ])

    return tfidf_preprocessor

def main():
    train = read_data('data/external/train.tsv')
    validation = read_data('data/external/validation.tsv')
    test = pd.read_csv('data/external/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    
    words_counts = get_words_count(X_train)
    tags_counts = get_tags_count(y_train)

    # Create bow preprocessor 
    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}

    bow_preprocessor = create_bow_pipeline(words_to_index=WORDS_TO_INDEX, dict_size=DICT_SIZE)

    # Create tfidf preprocessor
    tfidf_preprocessor = create_tfidf_pipeline()

    # Transform data
    X_train_mybag = bow_preprocessor.fit_transform(X_train)
    X_val_mybag = bow_preprocessor.transform(X_val)
    X_test_mybag = bow_preprocessor.transform(X_test)
    
    X_train_tfidf = tfidf_preprocessor.fit_transform(X_train)
    X_val_tfidf = tfidf_preprocessor.transform(X_val)
    X_test_tfidf = tfidf_preprocessor.transform(X_test)

    # Extract tfidf vectorizer vocab
    tfidf_vectorizer: TfidfVectorizer = tfidf_preprocessor['tfidf_vectorizer']
    tfidf_vocab = tfidf_vectorizer.vocabulary_

    # Save preprocessors
    if not os.path.exists('models/preprocessors'):
        os.makedirs(os.getcwd() + '/models/preprocessors', exist_ok=True)

    dump(bow_preprocessor, "models/preprocessors/bow_preprocessor.joblib")
    dump(tfidf_preprocessor, "models/preprocessors/tfidf_preprocessor.joblib")

    # Save preprocessed Data
    if not os.path.exists('data/processed'):
        os.makedirs(os.getcwd() + '/data/processed', exist_ok=True)

    dump(X_train_mybag, 'data/processed/preprocessed_x_train_mybag.joblib')
    dump(X_val_mybag, 'data/processed/preprocessed_x_val_mybag.joblib')
    dump(X_test_mybag, 'data/processed/preprocessed_x_test_mybag.joblib')

    dump(X_train_tfidf, 'data/processed/preprocessed_x_train_tfidf.joblib')
    dump(X_val_tfidf, 'data/processed/preprocessed_x_val_tfidf.joblib')
    dump(X_test_tfidf, 'data/processed/preprocessed_x_test_tfidf.joblib')
    dump(tfidf_vocab, 'data/processed/tfidf_vocab.joblib')

    # Save raw data
    if not os.path.exists('data/raw'):
        os.makedirs(os.getcwd() + '/data/raw', exist_ok=True)
        
    dump(X_train, 'data/raw/X_train.joblib')
    dump(X_val, 'data/raw/X_val.joblib')
    dump(y_train, 'data/raw/y_train.joblib')
    dump(y_val, 'data/raw/y_val.joblib')
    dump(tags_counts, 'data/raw/tags_counts.joblib')
    dump(WORDS_TO_INDEX, 'data/raw/WORDS_TO_INDEX.joblib')


if __name__ == "__main__":
    main()
