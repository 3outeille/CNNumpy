mport numpy as np

def tanh(x, constA = 1.7159):
    """
        Apply tanh function to x.         

        Parameters:
        -'x': input tensor.
        -'constA': constant value.
    """
    return constA * np.tanh(x)

def dTanH(x):
    """
        Apply derivative of tanh function to x.

        Parameters:
        -'x': input tensor.
    """
    return 1 - np.power(np.tanh(x), 2)

def softmax(x):
    """
        Compute softmax values for each sets of scores in x.
        
        Parameters:
        -'x': input vector.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def initializeFilter(n_f, f, n_C):
    """
        Initialize parameters.
            
        Parameters:
        -'n_f': number of filters.
        -'f': size of filters.
        -'n_C': number of channels.

        Returns:
        -'F': returns n_f weight of shape (f, f, n_C) 
        -'b': returns n_f bias of shape (1, 1, n_C)
   """
    
    F = np.random.randn(n_f, f, f, n_C)
    b = np.random.randn(n_f, 1, 1, n_C)    

    return F, b 

def initializeParamFc(x, y):
    """      
        Initialize parameters for fully connected layer.
        
        Parameters:
        -'x': row.
        -'y': column.
        
        Returns:
        -'W_fc': fully connected output weight of size (x, y).
        -'b_fc': fully connected output bias of size (x, 1).
    """
    W_fc = np.random.randn(x, y)
    b_fc = np.random.randn(x, 1)

    return W_fc, b_fc

def costFunction(yHat, y):
    """
        Compute the error between yHat and y using cross entropy loss function.

        Parameters:
        -'yHat': prediction.
        -'y': expected output.

        Returns:
        -'deltaL': error.
    """
    #Softmax.
    yHat = softmax(yHat) 

    #Cross entropy loss.
    deltaL = (1/y.shape(0)) * np.sum(y * np.log(yHat) + (1 - y) * np.log(1 - yHat))

    deltaL = np.squeeze(deltaL) #It turns [[1]] into 1.

    return deltaL

