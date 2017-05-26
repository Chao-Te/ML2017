import numpy as np
import sys

from hw5_utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json



testFile = sys.argv[1]
resFile = sys.argv[2]
tagsFile = './tags.pickle'
MAX_SEQUENCE_LENGTH = 313
nb_word = 50000
weights_name = './best'
arc_name = './hw5_architecture'
iFile = './index_mapping'

X_test = readTest(testFile)
tags = readTags(tagsFile)

tokenizer = readIndex(iFile)
sequences = tokenizer.texts_to_sequences(X_test)
X=pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

model = readModel(weights_name, arc_name)

  
Y_predict = model.predict(X, batch_size = 128)

res = np2tags(Y_predict , tags)

saveResult(resFile, res)
