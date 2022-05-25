"""
Class for Custom BOW Transformer
"""

import numpy as np

from scipy import sparse as sp_sparse
from sklearn.base import BaseEstimator, TransformerMixin


class BowVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom Bow Vectorizer class.
    """

    def __init__(self, words_to_index, dict_size):
        """
        Initialize BOW transformer.
        """
        self.dict_size = dict_size
        self.words_to_index = words_to_index

    def fit(self):
        """
        Transformer needs to implement fit, returns self.
        """
        return self

    def transform(self, X):
        """
        Transform data using my bag of words function.
        """
        return sp_sparse.vstack([sp_sparse.csr_matrix(self.my_bag_of_words(text)) for text in X])

    def my_bag_of_words(self, text):
        """
            text: a string
            dict_size: size of the dictionary

            return a vector which is a bag-of-words representation of 'text'
        """
        result_vector = np.zeros(self.dict_size)

        for word in text.split():
            if word in self.words_to_index:
                result_vector[self.words_to_index[word]] += 1
        return result_vector
