import numpy as np

from scripts.cleanup import CleanUpText
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# NLP
import spacy
import string
import nltk
from nltk.corpus import stopwords


class SFilterTransformer:
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        pass
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def transform(self, X, y=None):
        # Predict where is a spam
        predictions = self.model.predict(X)
        # Choose idx of non-spam reviews
        idx = np.array([i for i in range(predictions.shape[0])])[predictions == 0]
        return predictions[idx]