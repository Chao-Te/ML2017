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
from hw3_utils import *


train_csv='../data/train.csv'

num_classes=7
batch=256
num_epo=800
use_dropout=False
dropout_rate=0.4
use_norm=False
load_w=False

modelName='hw3_bonus_model.h5'
weights_name='hw3_bonus_weights.h5'
arc_name='hw3_bonus_arc'
testFile='../data/test.csv'
res='prediction_bonus.csv'

adam=optimizers.Adam(lr=0.000001,decay=1e-3)
rmsprop=optimizers.RMSprop(lr=0.00001,decay=1e-2)
opt=adam

X_train,Y_labeled=readTrain(train_csv)
Y_labeled = np_utils.to_categorical(Y_labeled, num_classes)

X_tmp=X_train[:25000]
Y_labeled=np.copy(Y_labeled)

X_train=X_train[25000:]
Y_train=np.copy(Y_labeled[25000:])

X_candidate=[]

for i in range(10):
    X_candidate.append(X_tmp[i*2500:(i+1)*2500])
    

keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

model=model_build(load_w,weights_name,use_dropout,dropout_rate,opt)

datagen = ImgGen()
datagen.fit(X_train)
history=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),
                    steps_per_epoch=len(X_train)//batch, epochs=num_epo)

seveAccuracy(history,'acc')
saveLoss(history,'Loss')

for i in  range(10):
    print('i=',i)
    Y=model.predict(X_candidate[i], batch_size=256)
    Y=np.argmax(Y,axis=1)
    Y = np_utils.to_categorical(Y, num_classes)
    X_train=np.concatenate((X_train,X_candidate[i] ), axis=0)
    print(Y_train.shape)
    print(Y.shape)
    Y_train=np.concatenate((Y_train,Y), axis=0)
    
    h=model.fit(X_train,Y_train,batch_size=batch,epochs=80)
    seveAccuracy(h,'acc'+str(i))
    saveLoss(h,'Loss'+str(i))
                    
####test
Y_labeled=np.argmax(Y_labeled,axis=1)
Y_tmp=model.predict(X_train, batch_size=256)
Y_tmp=np.argmax(Y_tmp,axis=1)
num=0
for i in range (len(Y_tmp)):
    if Y_tmp[i] != Y_labeled[i]:
        num+=1
Y=[(num/len(Y_tmp))*100]        
writeResult(Y,'error.csv')




X_test=readTest(testFile)
Y_test=model.predict(X_test, batch_size=256)
Y_test=np.argmax(Y_test,axis=1)
writeResult(Y_test,res)

                    
saveModel(model,modelName)
saveMosel_sep(model,weights_name,arc_name)
