import os
import csv
import keras
import numpy as np
import math
import sys
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
from keras import backend as K
from hw3_utils import *

def main():
    #####if you want to run my code please modify the path for test.csv########
    test_csv='./data/test.csv'
    
    json_file = open('hw3_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_classifier = model_from_json(loaded_model_json)
    emotion_classifier.load_weights('hw3_model_weights.h5')
    
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[0:])

    input_img = emotion_classifier.input
    name_ls = ['conv64_1','conv64_2','conv_128_1','conv128_2','conv256_1','conv256_2','conv512_1','conv512_2']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    private_pixels=readTest(test_csv)
    print(len(private_pixels))
    print(private_pixels.shape)
    private_pixels = [private_pixels[i].reshape((1, 48, 48, 1))
                      for i in range(len(private_pixels))]
    choose_id = 100
    photo = private_pixels[choose_id]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = 32
        if nb_filter<=128:
            for i in range(nb_filter):
                ax = fig.add_subplot(4, 8, i+1)
                ax.imshow(im[0][0, :, :, i], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
            img_path = './res_Q5_2'
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))
        else :
            for j in range(nb_filter//128):
                for i in range(j*128,(j+1)*128):
                    ax = fig.add_subplot(128/16, 16, i+1-j*128)
                    ax.imshow(im[0][0, :, :, i], cmap='BuGn')
                    plt.xticks(np.array([]))
                    plt.yticks(np.array([]))
                    plt.tight_layout()
                fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
                img_path = './res_Q5_2'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                fig.savefig(img_path+'/'+name_ls[cnt]+'_'+str(j)+'.png')
                
main()