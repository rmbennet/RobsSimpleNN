import numpy as np

data    =   [   [0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ]

data = np.array(data)

X_train =   data[:,0:2]
y_train =   data[:,-1].reshape((-1,1))
