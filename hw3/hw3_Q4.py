import csv
import keras
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


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
import keras.backend as K
from hw3_utils import*


########if you want to run my code please modify path of train.csv and test.csv#########
train_csv='./data/train.csv'
test_csv='./data/test.csv'


res='./res_Q4.csv'
num_classes=7



####load model#####
model_arch='hw3_architecture.json'
json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_classifier = model_from_json(loaded_model_json)
emotion_classifier.load_weights('hw3_model_weights.h5')


res_img='./res_img/'
private_pixels=readTest(test_csv);


##writeResult(Y_test,res)

private_pixels = [private_pixels[i].reshape((1, 48, 48, 1))
                      for i in range(len(private_pixels))]  
                      
input_img = emotion_classifier.input

#class0 =4,25
#class1=206,244
#class2=238,727
#class3=299,939
#class4=635,827
#class5=1011,1211
#class6=239,459
img_ids = [4,25,206,244,238,727,299,939,635,827,1011,1211,239,459]

for idx in img_ids:

    val_proba = emotion_classifier.predict(private_pixels[idx])
    pred = val_proba.argmax(axis=-1)
    print('idx=',idx,':',pred)
    target = K.mean(emotion_classifier.output[:, pred])
    
    grads = K.gradients(target, input_img)[0]
    #grads=grads/(K.sqrt(K.mean(K.square(grads))) + 1e-5)
    fn = K.function([input_img, K.learning_phase()], [grads])

    print(grads.shape)
    
    heatmap = fn([private_pixels[idx], 0])[0].reshape(2304)
    ######I have modify thres to get different in my paper
    thres = 0.5
    #####
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap.reshape(48, 48)   
    #print(dtype(heatmap))
    #heatmpa=K.eval(heatmap)
    '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
    '''
    see = private_pixels[idx].reshape(48, 48)
    see[np.where(heatmap <= thres)] = np.mean(see)
    
    
    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig(res_img+'color'+str(idx)+'_'+str(idx)+'.png', dpi=100)
    
    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig(res_img+'gray'+str(idx)+'_'+str(pred)+'.png', dpi=100)
K.clear_session()



