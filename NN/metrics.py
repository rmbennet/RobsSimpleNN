import numpy as np

def BCE(y_true, y_pred):
    """
    BCE : Binary Cross Entropy Loss
    computed using numpy arrays
    
    (See readme for definition of BCE)

    inputs : y_true, y_pred : numpy arrays of shape (n_samples, 1)

    outputs : loss : float
    """

    #using np.mean is more compact and faster:


    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred):

    """
    accuracy : computes the accuracy of the model
    inputs : y_true, y_pred : numpy arrays of shape (n_samples, 1)

    outputs : accuracy : float
    """
    #using np.mean is more compact and faster:
    return np.mean(y_true == y_pred)