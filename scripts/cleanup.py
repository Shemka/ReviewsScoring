import numpy as np
from tqdm import tqdm

# NLP
import spacy
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# This class was constructed especialy for sklearn.Pipeline
class CleanUpText:
    def __init__(self):
        # NLP HELPERS
        self.stopwords = stopwords.words('english')
        self.nlp = spacy.load("en_core_web_sm")
        self.punctuations = string.punctuation

    # Return True or False
    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def fit(self, docs):
        pass
    
    def transform(self, docs):
        nlp = self.nlp
        stopwords = self.stopwords
        punctuations = self.punctuations

        texts = []
        cuts = {
            "'s": ' is',
            "'m": ' am',
            "'ll": ' will',
            "'re": ' are',
            "'d": ' would',
            "'ve": ' have'
        }
        for doc in docs:
            for key in cuts.keys():
                if key in doc:
                    doc = doc.replace(key, cuts[key])

            doc = nlp(doc, disable=['parser', 'ner', 'tagger'])
            
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [cuts[tok] if tok in cuts.keys() else tok for tok in tokens]
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations and not self.hasNumbers(tok)]
            texts.append(' '.join(tokens))
        return texts

    def fit_transform(self, docs, y=None):
        nlp = self.nlp
        stopwords = self.stopwords
        punctuations = self.punctuations

        texts = []
        cuts = {
            "'s": ' is',
            "'m": ' am',
            "'ll": ' will',
            "'re": ' are',
            "'d": ' would',
            "'ve": ' have'
        }
        for doc in tqdm(docs):
            for key in cuts.keys():
                if key in doc:
                    doc = doc.replace(key, cuts[key])

            doc = nlp(doc, disable=['parser', 'ner', 'tagger'])
            
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [cuts[tok] if tok in cuts.keys() else tok for tok in tokens]
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations and not self.hasNumbers(tok)]
            texts.append(' '.join(tokens))
        return texts