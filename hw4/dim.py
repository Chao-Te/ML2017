import numpy as np

import math

import pickle
import csv
import sys
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors

from dim_utils import *



def write_result(resFile,r):
    f=open(resFile,'w')
    wf=csv.writer(f)
    wf.writerow(['SetId','LogDim'])
    for i in range (len(r)):
        wf.writerow([i,r[i]])
    f.close()
    

resFile=sys.argv[2]
model_name='model.pickle'
data_path=sys.argv[1]

file = open(model_name,'rb')
svr = pickle.load(file)
file.close()

testdata = np.load(data_path)

test_X = []

for i in range(200):
    print('i=',i)
    data = testdata[str(i)]

    vs = get_eigenvalues(data)

    test_X.append(vs)



test_X = np.array(test_X)

pred_y = svr.predict(test_X)
print('predict over')
print(pred_y[0])
for i in range(len(pred_y)):
    pred_y[i]=math.log(pred_y[i])

write_result(resFile,pred_y)
