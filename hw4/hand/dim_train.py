import numpy as np

import math

import pickle

from sklearn.svm import LinearSVR as SVR

from dim_utils import *

# Train a linear SVR


npzfile = np.load('hand_data.npz')
model_name='model.pickle'

X = npzfile['X']

y = npzfile['y']



# we already normalize these values in gen.py

# X /= X.max(axis=0, keepdims=True)



svr = SVR(C=1)

svr.fit(X, y)

with open(model_name, 'wb') as pickle_file:
   pickle.dump(svr,pickle_file)



print(svr.get_params())# to save the parameters

# svr.set_params() to restore the parameters



# predict