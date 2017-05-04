#score=69.657%

import csv
import keras
import numpy as np
import math
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import optimizers
from keras import regularizers
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def readTrain(train_csv):

    csvfile=open(train_csv)
    X_train=[]
    Y_train=[]
    first_train=True
    for row in csv.reader(csvfile,delimiter=','):
        if first_train:
            first_train=False
        else :
            
            Y_train.append(row[0])
            tmp=row[1].split(' ')
            
            X_train.append(np.reshape(np.array(tmp),(48,48,1)))
            
    Y_train=np.array(Y_train).astype(float)        
    X_train=np.array(X_train).astype(float)
    print ('size of X_train=',len(X_train))
    print ('shape X_train[0]',X_train[0].shape)
    Y_train=np.array(Y_train)
    print ('size of Y_train',len(Y_train))
    return X_train,Y_train
    
#main


train_csv=sys.argv[1]

num_classes=7
batch=256
num_epo=500        
use_dropout=False
dropout_rate=0.2
use_norm=False


adam=optimizers.Adam(lr=0.0001,decay=1e-3)
rmsprop=optimizers.RMSprop(lr=0.000001,decay=1e-1)
opt=adam


X_train,Y_train=readTrain(train_csv)
Y_train = np_utils.to_categorical(Y_train, num_classes)


keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)


model = Sequential()

#48*48

#####first#####
model.add(Convolution2D(64,(3,3),input_shape=(48,48,1),padding='same',name='conv64_1'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
model.add(Convolution2D(64,(3,3),padding='same',name='conv64_2'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))


model.add(MaxPooling2D(2,2))#maxpooling



#####second#####
model.add(Convolution2D(128,(3,3),padding='same',name='conv_128_1'))
#model.add(BatchNormalization())
model.add(Activation('relu'))

if use_dropout:
    model.add(Dropout(rate=dropout_rate))

model.add(Convolution2D(128,(3,3),padding='same',name='conv128_2'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
           
model.add(MaxPooling2D(2,2))#maxpooling


#####third#####
model.add(Convolution2D(256,(3,3),padding='same',name='conv256_1'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
model.add(Convolution2D(256,(3,3),padding='same',name='conv256_2'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
    
model.add(MaxPooling2D(2,2))#maxpooling



#####forth#####
model.add(Convolution2D(512,(3,3),padding='same',name='conv512_1'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
model.add(Convolution2D(512,(3,3),padding='same',name='conv512_2'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
 
model.add(MaxPooling2D(2,2))#maxpooling



######fully connected#####
model.add(Flatten())

#####FC1#####
model.add(Dense(4096,name='fc_1'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
#####FC2#####    
model.add(Dense(4096,name='fc_2'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
if use_dropout:
    model.add(Dropout(rate=dropout_rate))
    
#####softmax#####    
model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()


model.load_weights('hw3_model_weights.h5')


model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


#####image augmentation#####
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)


history=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),
                    steps_per_epoch=len(X_train)//batch, epochs=num_epo)

 





##model.save_weights('hw3_model_weights.h5')
##
##model_json = model.to_json()
##with open('hw3_architecture'+'.json', 'w') as json_file:
##    json_file.write(model_json)
