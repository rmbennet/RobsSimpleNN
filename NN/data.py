import numpy as np


def split_data(input_matrix):
    ''' 
    assuming a 2d numpy array where the
    last column is the label.
    
    returns: X_train , y_train
    y_train is reshaped to be a column vector
    '''

    return input_matrix[:,:-1], input_matrix[:,-1].reshape((-1,1))

d =  {

    'AND'   :   np.array    ([
                            [0, 0, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 1, 1]
                            ]),

    'OR'    :   np.array    ([
                            [0, 0, 0],
                            [0, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]
                            ]),

    'NOT'   :   np.array    ([
                            [0, 1],  # NOT 0 -> 1
                            [1, 0]   # NOT 1 -> 0
                            ])
    }
