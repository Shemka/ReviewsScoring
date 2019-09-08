import numpy as np

from scripts.cleanup import CleanUpText
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scripts.storing import load_model

# NLP
import spacy
import string
import nltk
from nltk.corpus import stopwords


class SFilterTransformer:
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def fit(self, X, y=None):
        pass
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def transform(self, X, y=None):
        # Predict where is a spam
        predictions = self.model.predict(X)
        # Choose idx of non-spam reviews
        idx = np.array([i for i in range(predictions.shape[0])])[predictions == 0]
        return np.array(X)[idx].tolist()