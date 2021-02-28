import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import load_model

vocab_size = 100000

max_len = 100

reviews_df = pd.read_csv("reviews.csv")

tokenizer = Tokenizer()

X_all = np.array(reviews_df['Review'])
Y_all = np.array(reviews_df['Label']) - 1

tokenizer.fit_on_texts(X_all)
X_all = tokenizer.texts_to_sequences(X_all)

X_all = pad_sequences(X_all, maxlen=max_len)

X_train = X_all[:80000]
Y_train = Y_all[:80000]

X_test = X_all[80000:]
Y_test = Y_all[80000:]

model = load_model("model.h5")

model.evaluate(X_test, Y_test)
