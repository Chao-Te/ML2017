import csv
import keras
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import optimizers
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from sklearn.metrics import confusion_matrix

   
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
            
    Y_train=np.array(Y_train).astype(int)        
    X_train=np.array(X_train).astype(float)
    print ('size of X_train=',len(X_train))
    print ('shape X_train[0]',X_train[0].shape)
    Y_train=np.array(Y_train)
    print ('size of Y_train',len(Y_train))
    return X_train,Y_train
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

num_classes=7
#################if you want to run my code please modify the directory of train.csv##########    
train_csv='./data/train.csv'

model_arch='hw3_architecture.json'

X_train,Y_train=readTrain(train_csv)

X_val=X_train[25000:]
Y_val=Y_train[25000:]


json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('hw3_model_weights.h5')


predictions = model.predict_classes(X_val)

print(type(Y_val))
print(type(predictions))

conf_mat = confusion_matrix(Y_val,predictions)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig('confu.png')
plt.clf()