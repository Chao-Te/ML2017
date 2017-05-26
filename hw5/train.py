import numpy as np
import json
from hw5_utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping

trainFile = './data/train_data.csv'
testFile = './data/test_data.csv'
resFile = './result/prediction.csv' 
tname = './result/tags.pickle'
pre_model = './glove.6B.200d.txt'
iFile = './result/index_mapping'



nb_word = 50000
EMBEDDING_DIM = 200
batch = 128
num_epo= 200


load_w = False
use_dropout = True
dropout_rate = 0.5
split_ratio = 0.1
weights_name = './result/hw5_weight'
arc_name = './result/hw5_architecture'

#opt = optimizers.Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
opt = optimizers.RMSprop(lr=0.001,clipvalue=0.5)

X_train, Y_train, tags = readTrain(trainFile)
X_test = readTest(testFile)
print(len(X_train))
print (len(X_test))

labels = tags2np(Y_train,tags)
saveTags(tname,tags)

X = X_test[:]# for word embedding
X.extend(X_train[:])
#X=X_train[:]



embeddings_index = readGolve(pre_model)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index    
saveIndex(iFile, tokenizer)    
    
data = pad_sequences(sequences)
MAX_SEQUENCE_LENGTH = len(data[0])

print (MAX_SEQUENCE_LENGTH)

X_train = data [len(X_test):]
Y_train = labels
X_test = data [:len(X_test)]

(X_train,Y_train),(X_val,Y_val) = split_data(X_train,Y_train,split_ratio)
print(data.shape)
'''
network 
'''


num_words =  len(word_index)+1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


model = Sequential()
model.add(embedding_layer)

model.add(GRU(256, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))
model.add(GRU(256, dropout=dropout_rate, recurrent_dropout=dropout_rate))
   
model.add(Dense(256, activation = 'elu'))
if use_dropout:
        model.add(Dropout(rate=dropout_rate))
        
model.add(Dense(128, activation = 'elu'))
if use_dropout:
        model.add(Dropout(rate=dropout_rate))        

model.add(Dense(64, activation = 'elu'))
if use_dropout:
        model.add(Dropout(rate=dropout_rate))        
        
model.add(Dense(len(tags)))
model.add(Activation('sigmoid'))

if load_w:
        model.load_weights(weights_name)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=[f1_score])
#earlystopping = EarlyStopping(monitor='val_f1_score', patience = 15, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='./result/best.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch,epochs=num_epo,callbacks=[ checkpoint])
#earlystopping,



saveModel(model,weights_name,arc_name)

#Y_predict = model.predict(data[:len(X_test)], batch_size = 128)

#res = np2tags(Y_predict , tags)

#saveResult(resFile, res)
