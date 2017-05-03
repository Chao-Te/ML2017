import csv
import keras
import numpy as np
import math
import matplotlib.pyplot as plt


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
    
    
def seveAccuracy(history,imgname):
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(imgname+'.png')
    plt.clf()
    
def saveLoss(history,imgname):
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch') 
    plt.legend(['loss'], loc='upper left')
    plt.savefig(imgname+'.png')
    plt.clf()
    
    
def writeResult(Y_test,res):
    f=open(res,'w')
    wf=csv.writer(f)
    wf.writerow(['id','label'])
    for i in range (len(Y_test)):
        wf.writerow([i,Y_test[i]])
    f.close()


def readTest(test_csv):
    csvfile=open(test_csv)
    X_test=[]
    first=True
    
    for row in csv.reader(csvfile,delimiter=','):
        if first:
            first=False
        else:
            tmp=row[1].split(' ')
            X_test.append(np.reshape(np.array(tmp),(48,48,1)))
    
        
    X_test=np.array(X_test).astype(float)
    return X_test

def model_build(load_w,weights_name,use_dropout,dropout_rate,opt):
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
    model.add(Activation('relu'))
    
    if use_dropout:
        model.add(Dropout(rate=dropout_rate))
    
    model.add(Convolution2D(128,(3,3),padding='same',name='conv128_2'))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(rate=dropout_rate))
            
    model.add(MaxPooling2D(2,2))#maxpooling
    
    
    #####third#####
    model.add(Convolution2D(256,(3,3),padding='same',name='conv256_1'))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Convolution2D(256,(3,3),padding='same',name='conv256_2'))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(rate=dropout_rate))
        
        
    model.add(MaxPooling2D(2,2))#maxpooling
    
    
    
    #####forth#####
    model.add(Convolution2D(512,(3,3),padding='same',name='conv512_1'))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Convolution2D(512,(3,3),padding='same',name='conv512_2'))
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
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(rate=dropout_rate))
        
    #####softmax#####    
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    model.summary()
    
    if load_w:
        model.load_weights(weights_name)
    
    
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model
    
    
def saveModel(model,modelName):
    model.save(modelName)

    
def saveMosel_sep(model,weights_name,arc_name):
    model.save_weights(weights_name)

    model_json = model.to_json()    
    with open(arc_name+'.json', 'w') as json_file:
        json_file.write(model_json)
        
        
        
def ImgGen():
    tmp=ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
    return tmp