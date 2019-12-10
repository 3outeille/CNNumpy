import numpy as np
from utils import *

def fcFwd(fc, W, b):
    """
        Perfoms a forward propagation between 2 fully connected layers. 

        Parameters:
        -'fc': last convolved layer flatten in size.
        -'W': weight.
        -'b': bias.
        Returns:
        -'A_fc': fully connected layer.
    """
    A_fc = np.dot(W, fc) + b

    return A_fc

def convFwd(A_prev, W, b, p, s):
    """
        Perfoms a forward convolution.         
        
        Parameters:
        -'A_prev': Input from last pooling of shape (m, n_H_prev, n_W_prev, n_C_prev).
                         If first convolution, A_prev = X.
        -'W': weight at layer l of shape (n_f, f, f, n_C_prev).
        -'b': bias at layer l of shape (n_f, 1, 1, n_C_prev)
        -'p': input padding.
        -'s': input stride.

        Returns:
        -'A_conv': previous input convolved of shape (m, n_H, n_W, n_C)
    """

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    n_f, f, f, n_C_prev = W.shape
    n_f, f, f, n_C_prev = b.shape
    
    n_H = int((n_H_prev + 2 * p - f)/ s) + 1 #height.
    n_W = int((n_W_prev + 2 * p - f)/ s) + 1 #width.
    n_C = n_f #number of channels. 

    A_conv = np.zeros((m, n_H, n_W, n_C))

    for i in range(m): #For each image.
        
        for h in range(n_H): #Slide the filter vertically.
            h_start = h * s
            h_end = h_start + f
            
            for w in range(n_W): #Slide the filter horizontally.                
                w_start = w * s
                w_end = w_start + f

                A_conv[i] = np.sum(A_prev[i, h_start:h_end, w_start:w_end] * W[i]) + b[i]

    return A_conv

def actTanh(A_conv):
    """
        Apply activation function after convolving.
        
        Parameters:
        -'A_conv': output of convFwd() function.
        
        Returns:
        -'A_conv_act': tanh function applied on conv output.
    """
    A_conv_act = tanh(A_conv)

    return A_conv_act


def avgpoolFwd(A_conv_act, f, p, s):
    """
        Apply average pooling during forward propagation.

        Parameters: 
        -'A_conv_act': output of activationTanh()
        -'f': filter size.
        -'p': input padding.
        -'s': input stride.

        Returns:
        -'A_pool': A_conv_act squashed of shape (m, n_H, n_W, n_C).
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_conv_act.shape
  
    n_H = int((n_H_prev - f)/ s) + 1 #height.
    n_W = int((n_W_prev - f)/ s) + 1 #width.
    n_C = n_f #number of channels. 

    A_pool = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):

        for h in range(n_H): 
            h_start = h * s
            h_end = h_start + f

            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f
                
                A_pool[i] = np.mean(A_conv_act[i, h_start:h_end, w_start:w_end])

    return A_pool

