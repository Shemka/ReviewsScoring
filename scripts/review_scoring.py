import warnings
warnings.filterwarnings("ignore")

from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, Flatten, Conv1D, SpatialDropout1D, MaxPooling1D, Dense, GRU, LSTM, Dropout, BatchNormalization, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping
from scripts.storing import load_model as lm
from scripts.cleanup import CleanUpText

class ReviewScorer:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = lm(tokenizer_path)
        self.model = load_model(model_path)
    
    def predict(self, texts):
        texts = CleanUpText().transform(texts)
        seq = self.tokenizer.texts_to_sequences(texts)
        seq = pad_sequences(seq, maxlen=self.model.layers[0].input_shape[1])
        return self.model.predict(seq).reshape(-1)      