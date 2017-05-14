import numpy as np

import math

import pickle
import csv
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from dim_utils import *



def write_result(resFile,r):
    f=open(resFile,'w')
    wf=csv.writer(f)
    wf.writerow(['SetId','LogDim'])
    for i in range (len(r)):
        wf.writerow([i,r[i]])
    f.close()

def readImg():
    data=[]
    for i in range (1,482):
        imName='./hand/hand.seq'+str(i)+'.png'
        im = Image.open(imName)
        (h,w)=im.size
        h=32
        w=30
        im = im.resize( (h,w), Image.BILINEAR )
        im=np.array(im)
        data.append(np.reshape(im,h*w))
    data=np.array(data).astype(float)
    #min=np.amin(data)
    #max=np.amax(data)
    #data=data-min
    #data=data/(max-min)
    return data


resFile='prediction.csv'

model_name='model.pickle'
file = open(model_name,'rb')
svr = pickle.load(file)
file.close()

data = readImg()

test_X = []

vs = get_eigenvalues(data)

test_X.append(vs)



test_X = np.array(test_X)

pred_y = svr.predict(test_X)
print('predict over')
print(pred_y)

for i in range(len(pred_y)):
    pred_y[i]=math.log(pred_y[i])

write_result(resFile,pred_y)
