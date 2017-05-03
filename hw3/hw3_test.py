import csv
import keras
import numpy as np
import math
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import optimizers
from keras.models import model_from_json

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

def readTest_preprocess(test_csv):
    csvfile=open(test_csv)
    X_test=[]
    first=True
    
    for row in csv.reader(csvfile,delimiter=','):
        if first:
            first=False
        else:
            tmp=row[1].split(' ')
            
            tmp=np.array(tmp).astype(float)
            tmp-=np.mean(tmp)
            X_test.append(np.reshape(tmp,(48,48,1)))
    X_test=np.array(X_test).astype(float)
    return X_test    
    
    
    
    
def writeResult(Y_test,res):
    f=open(res,'w')
    wf=csv.writer(f)
    wf.writerow(['id','label'])
    for i in range (len(Y_test)):
        wf.writerow([i,Y_test[i]])
    f.close()
#main
testFile=sys.argv[1]
res=sys.argv[2]
model_arch='hw3_architecture.json'



X_test=readTest(testFile)

json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('hw3_model_weights.h5')

Y_test=model.predict(X_test, batch_size=100)
Y_test=np.argmax(Y_test,axis=1)
print(Y_test.shape)
print(Y_test[0])
writeResult(Y_test,res)


