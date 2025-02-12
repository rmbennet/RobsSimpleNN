import numpy as np

#activation function
from scipy.special import softmax


class Linear():
    def __init__(self, num_features=2,num_neurons=2):

        ''' 
        2 neurons each having:
            - 2 weights
            - 1 bias

        Note the structure here when doing the linear transformation
        the number biases is equal to num_neurons
        '''

        #---------------------------------------
        #RANDOMLY INITIALIZE WEIGHTS AND BIASES:
        #---------------------------------------

        #sample from the standard normal distribution
        #quantity of weights = num_inputs:
        #initialized as a column vector:
        self.weights = np.random.randn(num_neurons,num_features) *0.01
        self.bias = np.random.randn(num_neurons)


    def forward(self, inputs):

        ''' 
        Forward pass through a neuron.
        inputs : numpy array

        Not making this too dynamic since new:

        using the weights and biases previously defined, 
        perform a linear transformation on the input, then do a sigmoid
        activation on it
        '''
        #ensure inputs is a numpy array:
        self.inputs = np.array(inputs)
        #---------------------------------------
        #LINEAR TRANSFORMATION:
        #---------------------------------------
        y = (inputs@ self.weights + self.bias)
    
        #---------------------------------------
        #SOFTMAX ACTIVATION:
        #---------------------------------------
        
        #need to specify an axis for softmax!

        #clip the predictions to avoid numerical instability near log(0) and log(1)
        y = np.clip(np.array(y),1e-3,1-1e-3)
        
        y_pred = softmax(y,axis=1)

        return y_pred
    
    '''

    LEAVING BACKWARD PASS OMITTED FOR NOW
    GET FORWARD COMPLETED FIRST!
    def backward(self, y_true):
        d_output = self.output - y_true
        d_z = d_output * (self.output) * (1 - self.output)  # Gradient with respect to the pre-activation

        # Ensure that the shape of d_z matches the expected input size
        d_z = d_z.reshape(-1, 1)  # Reshaping d_z for compatibility in matrix multiplication

        # Calculate the gradient of the weights and bias
        d_weights = np.dot(self.inputs.T, d_z)  # Shape should align now
        d_bias = np.mean(d_z, axis=0)

        # Update weights and biases
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        # Return the gradient with respect to the inputs (to propagate further back)
        d_input = np.dot(d_z, self.weights.T)  # Propagate the gradient back
        return d_input

    '''