import numpy as np

from sklearn.neighbors import NearestNeighbors

from dim_utils import *





# generate some data for training

X = []

y = []

for i in range(60):

    dim = i + 1
    #, 20000,30000,40000, 50000,70000, 80000, 100000
    for N in range(4):
        
        layer_dims = [np.random.randint(60, 80), 960]

        data = gen_data(dim, layer_dims, 481).astype('float32')

        eigenvalues = get_eigenvalues(data)

        X.append(eigenvalues)

        y.append(dim)



X = np.array(X)

y = np.array(y)



np.savez('hand_data.npz', X=X, y=y)
