import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreparer(BaseEstimator, TransformerMixin):
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