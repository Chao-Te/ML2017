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


##  json_file = open('hw3_architecture.json', 'r')
##  loaded_model_json = json_file.read()
##  json_file.close()
##  model = model_from_json(loaded_model_json)
##  model.load_weights('hw3_model_weights.h5')
##  
##  model = load_model('hw3_model.h5')
##  layer_name = 'conv64_1'
##  layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
##  filters = np.arange(get_num_filters(model.layers[layer_idx]))
##  vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx))
##                for idx in filters]
##  
##  plt.figure()
##  plt.imshow(vis_images[0])
##  plt.savefig('test.png')
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    x=0
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data,0])
        input_image_data += grads_value * 1
        #x+=1
    img=input_image_data[0]
    
    img=img.reshape(48,48)
    
    return [img,loss_value]

def main():
    NUM_STEPS=20
    RECORD_FREQ=20
    nb_filter=32
    num_step=20
    json_file = open('hw3_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_classifier = model_from_json(loaded_model_json)
    emotion_classifier.load_weights('hw3_model_weights.h5')
    
    
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[0:])
    
    input_img = emotion_classifier.input

    name_ls = ['conv64_1','conv64_2','conv_128_1','conv128_2','conv256_1','conv256_2','conv512_1','conv512_2']
    
    
    collect_layers = [ layer_dict[name].output for name in name_ls ]
    print( 'in for loop')
    for cnt, c in enumerate(collect_layers):
        #print( 'in for loop')
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])
            
            ###
            "You need to implement it."
            filter_imgs[0].append(grad_ascent(num_step, input_img_data, iterate))
            ###

        for it in range(NUM_STEPS//RECORD_FREQ):
            if nb_filter<=128:
                fig = plt.figure(figsize=(14, 8))
                for i in range(nb_filter):
                    ax = fig.add_subplot(4, 8, i+1)
                    #print(filter_imgs[it][i][0].shape)
                    ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                    plt.xticks(np.array([]))
                    plt.yticks(np.array([]))
                    plt.xlabel('{:.3f}'.format(i))
                    plt.tight_layout()
                fig.suptitle('Filters of layer {} '.format(name_ls[cnt]))
                img_path = './res_Q5'
                fig.savefig(img_path+'/'+name_ls[cnt]+'.png')
            else:
                for j in range(nb_filter//128):
                    fig = plt.figure(figsize=(14, 8))
                    for i in range(j*128,(j+1)*128):
                        ax = fig.add_subplot(128/16, 16, i+1-j*128)
                        #print(filter_imgs[it][i][0].shape)
                        ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                        plt.xticks(np.array([]))
                        plt.yticks(np.array([]))
                        plt.xlabel('{:.3f}'.format(i))
                        plt.tight_layout()
                    fig.suptitle('Filters of layer {} '.format(name_ls[cnt]))
                    img_path = './res_Q5'
           
                    fig.savefig(img_path+'/'+name_ls[cnt]+'_'+str(j)+'.png')
main()