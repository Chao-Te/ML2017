import numpy as np
import csv
import pickle
import json
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K

def tags2np(Y_train,tags):
    labels=[]
    #print(tags)
    
    for y in Y_train :
        label=np.zeros(len(tags))
        for i in y :
            label[tags.index(i)]=1
        labels.append(label)
    labels=np.array(labels).astype(float)
    
    return labels
    
    
def readTrain(tFile):

    #csvfile = open(tFile,'r',encoding='ISO-8859-1')
    #csvfile.readline()
    X_train = []
    Y_train = []
    tags = []
    
    i=0
    with open(tFile, encoding = 'utf8') as csvfile :
    
        for row in csvfile :
            if i!=0 :
                #print(row)
                row = row.split('"')
                row.pop(0)
                tag=(row.pop(0)).split(' ')
                for t in tag :# list all tags
                    if t not in tags:
                        tags.append(t)
                Y_train.append(tag)
                
                row = '"'.join(row)
                row = row [1:]
                X_train.append (row)
            else :
                print(row)
                i+=1
    return X_train,Y_train,tags
    
    
def readTest(testFile) :
    csvfile = open(testFile,'r',encoding='ISO-8859-1')
    csvfile.readline()
    X_test=[]
    
    for row in csvfile :
        row = row.split(',')
        row.pop(0)
        row = ','.join(row)
        X_test.append(row)
    
    return X_test
    
    
def saveResult(resFile,res) :
    f=open(resFile,'w')
    f.write('"id","tags"\n')
    for i in range (len(res)):
        f.write('"'+str(i)+'"'+','+'"'+res[i]+'"\n' )
    f.close()
    
    
def saveTags(tname,tags) :
    with open(tname, 'wb') as pickle_file:
        pickle.dump(tags,pickle_file)
    
def readTags (tname) :
    file = open(tname,'rb')
    tags = pickle.load(file)
    file.close()
    return tags

def saveModel(model,weights_name,arc_name):
    model.save_weights(weights_name+'.h5')

    model_json = model.to_json()    
    with open(arc_name+'.json', 'w') as json_file:
        json_file.write(model_json)
        
        
def readModel(weights_name, arc_name) :

    json_file = open(arc_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_name+'.h5')
    
    return model

def np2tags(Y_predict, tags) :
    res=[]
    
    for i in range(len(Y_predict)) :
        res_tags=[]
        for j in range(len(tags)) :
            
            if Y_predict[i][j] >=0.4 :
                res_tags.append(tags[j])
                Y_predict[i][j] = 1
            else :
                Y_predict[i][j] = 0
        ##if len(res_tags) == 0 :
        ##    res_tags.append(tags[np.argmax(Y_predict[i])])
        res.append(' '.join(res_tags))
        
    return res


def readGolve(gFile) :
    embeddings_index = {}
    with open(gFile, encoding = 'utf8') as f :
        #f.readline()
        for line in f:
            values = line.split(' ')
            word = values[0]
          
            coefs = np.asarray(values[1:], dtype='float32')
            
            embeddings_index[word] = coefs
    
    return embeddings_index

def saveIndex(iFile, word_index) :
    with open(iFile+'.pickle', 'wb') as pickle_file:
        pickle.dump(word_index,pickle_file)
        
def readIndex(iFile):
    file = open(iFile+'.pickle','rb')
    dic = pickle.load(file)
    file.close()
    
    return dic
    
        
'''
for loss        
'''     
##from keras import backend as K
##
##def precision(y_true, y_pred):
##    """Precision metric.
##    Only computes a batch-wise average of precision.
##    Computes the precision, a metric for multi-label classification of
##    how many selected items are relevant.
##    """
##    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
##    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
##    precision = true_positives / (predicted_positives + K.epsilon())
##    return precision
##
##
##def recall(y_true, y_pred):
##    """Recall metric.
##    Only computes a batch-wise average of recall.
##    Computes the recall, a metric for multi-label classification of
##    how many relevant items are selected.
##    """
##    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
##    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
##    recall = true_positives / (possible_positives + K.epsilon())
##    return recall
##
##
##def F1(y_true,y_pred) :
##    beta = 1
##    p = precision(y_true, y_pred)
##    r = recall(y_true, y_pred)
##    bb = beta ** 2
##    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
##    return fbeta_score
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))
    
    
def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)
