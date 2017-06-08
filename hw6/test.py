import pandas
import numpy as np
import keras
import csv
import sys
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.layers import Input, Flatten, add, dot, Add, Dot, Concatenate, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.models import load_model, Model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping


def readModel(weights_name, arc_name) :

    json_file = open(arc_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_name+'.h5')
    
    return model

def savePredict(resfile, predict) :
    f = open (resfile, 'w')
    wf = csv.writer(f)
    wf.writerow(['TestDataID', 'Rating'])
    for i in range(len(predict)):
        wf.writerow([i+1, predict[i][0]])
    f.close()
dir_path = sys.argv[1]
res_path = sys.argv[2]
userfile = dir_path + 'users.csv'
trainfile = dir_path + 'train.csv'
moviefile = dir_path + 'movies.csv'
testfile = dir_path + 'test.csv'
predict_name = res_path
weights_name = './models/best'
network_name = './models/hw6_arc'

users = pandas.read_csv(userfile, sep='::',engine='python')
training = pandas.read_csv(trainfile, sep=',')
movies = pandas.read_csv(moviefile ,sep='::',engine='python')
testing = pandas.read_csv(testfile, engine='python', sep=',')


test_id = testing.TestDataID
test_movie_id = testing.MovieID.values
test_user_id = testing.UserID.values

model = readModel(weights_name,network_name)

y_pred = model.predict([test_movie_id,test_user_id])

savePredict(predict_name, y_pred)
