import numpy as np

from scipy import sparse as sp_sparse
from sklearn.base import BaseEstimator, TransformerMixin

class BowVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom BOW Transformer
    """
    def __init__(self, words_to_index=[], dict_size=5000):
        self.dict_size = dict_size
        self.words_to_index = words_to_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
<<<<<<< HEAD
        return sp_sparse.vstack([sp_sparse.csr_matrix(self.my_bag_of_words(text)) for text in X])

    def my_bag_of_words(self, text):
=======
        return sp_sparse.vstack([sp_sparse.csr_matrix(self.my_bag_of_words(text, self.words_to_index, self.dict_size)) for text in X])

    def my_bag_of_words(self, text, words_to_index, dict_size):
>>>>>>> 65406bb (refactor processing into pipelines)
        """
            text: a string
            dict_size: size of the dictionary
            
            return a vector which is a bag-of-words representation of 'text'
        """
<<<<<<< HEAD
        result_vector = np.zeros(self.dict_size)
        
        for word in text.split():
            if word in self.words_to_index:
                result_vector[self.words_to_index[word]] += 1
=======
        result_vector = np.zeros(dict_size)
        
        for word in text.split():
            if word in words_to_index:
                result_vector[words_to_index[word]] += 1
>>>>>>> 65406bb (refactor processing into pipelines)
        return result_vector