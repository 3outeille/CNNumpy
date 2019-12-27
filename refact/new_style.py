import numpy as np

class Conv():
    """
        Convolutional layer.
    """
    
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding

        #Initialize Weight/bias.
        self.W = {'val': np.random.randn(self.n_F, self.f, self.f, self.n_C),
                  'grad': np.zeros((self.n_F, self.f, self.f, self.n_C))}
        self.b = {'val': np.random.randn(self.n_F, 1, 1, self.n_C),
                  'grad': np.zeros((self.n_F, 1, 1, self.n_C))}
        
        self.cache = None

    def forward(self, X):
        """
            Performs a forward convolution.
            Parameters:
            - A_prev: Last layer of shape (m, n_H_prev, n_W_prev, n_C_prev).
                     If first convolution, A_prev = X.
            Returns:
            - A_conv: previous layer convolved.
        """
        m, n_H_prev, n_W_prev, n_C_prev = X.shape

        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        n_C = self.n_F

        #A_conv = np.zeros((m, n_H, n_W, n_C))
        out = np.zeros((m, n_H, n_W, n_C))
        
        #print('A_conv:' + str(A_conv.shape))
        #print('A_prev:' + str(A_prev.shape))
        #print('W:' + str(self.W['val'].shape))
        #print('b:' + str(self.b['val'].shape))

        for i in range(m): #For each image.
            
            for h in range(n_H): #Slide the filter vertically.
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    for c in range(n_C): #For each channel.
                        #print('--------')
                        #print(A_prev[i, h_start:h_end, w_start:w_end].shape) 
                        #print(self.W['val'][c, :, :, :].shape) 
                        #print(self.b['val'][c, :, :, :].shape) 
                         
                        out[i, h, w, c] = np.sum(X[i, h_start:h_end, w_start:w_end] 
                                        * self.W['val'][c, ...] + self.b['val'][c, ...])
        self.cache = X

        return out 

    def backward(self, dout):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - A_prev_error: error from previous layer.
            Returns:
            - deltaL: error of the current convolutional layer.
        """
        X = self.cache
        
        m, n_H, n_W, n_C = X.shape
        _, n_H_dout, n_W_dout, n_C_dout = dout.shape

        
        #Distribute error in weight (compute gradient of weight).
        for i in range(m): #For each examples.
            
            for c in range(n_C_dout): #Take one channel and duplicate it n_C time along depth axis.
                
                dout_tile = np.repeat(dout[i, :, :, c][..., np.newaxis], repeats = n_C , axis = 2)
                #print('dout_tile: ' + str(dout_tile.shape))

                for h in range(self.f):
                    h_start = h * self.s
                    h_end = h_start + n_H_dout

                    for w in range(self.f):
                        w_start = w * self.s
                        w_end = w_start + n_W_dout

                        self.W['grad'][c, ...] = np.sum(X[i, h_start:h_end, w_start:w_end, :] * dout_tile)
                   
        #Distribute error in bias (compute gradient of bias).
        for i in range(self.n_F):
            self.b['grad'][i, ...] = np.sum(dout[..., i])
 
                
        #Compute padding needed to perform a full convolution.
        pad = int((1/2) * (self.f - n_H_dout + (n_H - 1) / self.s))
        dout_pad = np.pad(dout, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        W_rot = np.rot90(np.rot90(self.W['val']))
        dX = np.zeros(X.shape)

        for i in range(m): #For each examples.
            
            for c in range(self.n_C): #Take one weight channel and duplicate it n_C_dout time along depth axis.
                
                W_rot_tile = W_rot[:, :, :, c].reshape((W_rot.shape[1], W_rot.shape[2], W_rot.shape[0]))
                #W_rot_tile = np.repeat(W_rot[i, :, :, c][..., np.newaxis], repeats = n_C_dout , axis = 2)
                #print('W_rot_tile: ' + str(W_rot_tile.shape))
       
                for h in range(n_H):
                    h_start = h * self.s
                    h_end = h_start + self.f
                    
                    #print(h_start, h_end, end='\n\n')

                    for w in range(n_W):                
                        w_start = w * self.s
                        w_end = w_start + self.f
                        
                        #print(h, w, "dout_pad: " + str(dout_pad[i, h_start:h_end, w_start:w_end, :].shape))
                        #print("W_rot_tile: " + str(W_rot_tile.shape), end='\n\n')

                        dX[i, h, w, c] = np.sum(dout_pad[i, h_start:h_end, w_start:w_end, :] * W_rot_tile)

        return dX, self.W['grad'], self.b['grad']
        
        
class AvgPool():
    
    def __init__(self, filter_size, stride):
        self.f = filter_size
        self.s = stride
        self.cache = None

    def forward(self, X):
        """
            Apply average pooling on A_conv_act.
            Parameters:
            - A_conv_act: Output of activation function.
            Returns:
            - A_pool: A_conv_act squashed. 
        """
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        
        n_H = int((n_H_prev - self.f)/ self.s) + 1
        n_W = int((n_W_prev - self.f)/ self.s) + 1
        n_C = n_C_prev

        A_pool = np.zeros((m, n_H, n_W, n_C))
    
        for i in range(m): #For each image.
            
            for h in range(n_H): #Slide the filter vertically.
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    A_pool[i, h, w] = np.mean(X[i, h_start:h_end, w_start:w_end])
        
        self.cache = X

        return A_pool

    def backward(self, dout):
        """
            Distributes error through pooling layer.
            Parameters:
            - A_prev_error: Previous layer with the error.
            Returns:
            - A_pool_tmp: Temporary pooling layer with the error.
        """
        X = self.cache
        m, n_H, n_W, n_C = dout.shape
        dX = np.copy(X)        

        for i in range(m):

            for h in range(n_H):
                h_start = h * self.s
                h_end = h_start + self.f

                for w in range(n_W):
                    w_start = w * self.s
                    w_end = w_start + self.f

                    for c in range(n_C):
                        average = dout[i, h, w, c] / (n_H * n_W)
                        filter_average = np.full((self.f, self.f), average)
                        dX[i, h_start:h_end, w_start:w_end, c] += filter_average

        return dX


class Fc():

    def __init__(self, row, column):
        self.row = row
        self.col = column
        
        #Initialize Weight/bias.
        self.W = {'val': np.random.randn(self.row, self.col), 'grad': 0}
        self.b = {'val': np.random.randn(self.row, 1), 'grad': 0}
        
        self.cache = None

    def forward(self, fc):
        """
            Performs a forward propagation between 2 fully connected layers.
            Parameters:
            - fc: fully connected layer.
            Returns:
            - A_fc: new fully connected layer.
        """
        self.cache = fc
        A_fc = np.dot(self.W['val'], fc) + self.b['val']
        return A_fc

    def backward(self, deltaL):
        """
            Returns the error of the current layer and compute gradients.
            Parameters:
            - deltaL: error at last layer.
            Returns:
            - new_deltaL: error at current layer.
        """
        fc = self.cache
        m = fc.shape[0]
        
        #Compute gradient.
        self.W['grad'] = (1/m) * np.dot(deltaL, fc.T)
        self.b['grad'] = (1/m) * np.sum(deltaL, axis = 0)
        
        #Compute error.
        new_deltaL = np.dot(self.W['val'].T, deltaL) 
        #We still need to multiply new_deltaL by the derivative of the activation
        #function which is done in TanH.backward() only when we have to backpropagate
        #error through tanH.

        return new_deltaL, self.W['grad'], self.b['grad']
    

class AdamGD():

    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params

        #Initialize Momentum parameters.
        self.vdW1, self.vdb1 = np.zeros(self.params['W1'].shape), np.zeros(self.params['b1'].shape)
        self.vdW2, self.vdb2 = np.zeros(self.params['W2'].shape), np.zeros(self.params['b2'].shape)
        # self.vdW3, self.vdb3 = np.zeros(self.params['W3'].shape), np.zeros(self.params['b3'].shape)
        # self.vdW4, self.vdb4 = np.zeros(self.params['W4'].shape), np.zeros(self.params['b4'].shape)
        # self.vdW5, self.vdb5 = np.zeros(self.params['W5'].shape), np.zeros(self.params['b5'].shape)

        #Initialize RMSpop parameters.
        self.sdW1, self.sdb1 = np.zeros(self.params['W1'].shape), np.zeros(self.params['b1'].shape) 
        self.sdW2, self.sdb2 = np.zeros(self.params['W2'].shape), np.zeros(self.params['b2'].shape) 
        # self.sdW3, self.sdb3 = np.zeros(self.params['W3'].shape), np.zeros(self.params['b3'].shape) 
        # self.sdW4, self.sdb4 = np.zeros(self.params['W4'].shape), np.zeros(self.params['b4'].shape) 
        # self.sdW5, self.sdb5 = np.zeros(self.params['W5'].shape), np.zeros(self.params['b5'].shape) 


    def update_params(self, grads):
        """

        """
        #Momentum update.
        self.vdW1 = (self.beta1 * self.vdW1) + (1 - self.beta1) * grads['dW1'] 
        self.vdW2 = (self.beta1 * self.vdW2) + (1 - self.beta1) * grads['dW2']
        # self.vdW3 = (self.beta1 * self.vdW3) + (1 - self.beta1) * grads['dW3']
        # self.vdW4 = (self.beta1 * self.vdW4) + (1 - self.beta1) * grads['dW4']
        # self.vdW5 = (self.beta1 * self.vdW5) + (1 - self.beta1) * grads['dW5']
                
        self.vdb1 = (self.beta1 * self.vdb1) + (1 - self.beta1) * grads['db1']  
        self.vdb2 = (self.beta1 * self.vdb2) + (1 - self.beta1) * grads['db2'] 
        # self.vdb3 = (self.beta1 * self.vdb3) + (1 - self.beta1) * grads['db3'] 
        # self.vdb4 = (self.beta1 * self.vdb4) + (1 - self.beta1) * grads['db4'] 
        # self.vdb5 = (self.beta1 * self.vdb5) + (1 - self.beta1) * grads['db5'] 

        #RMSpop update.
        self.sdW1 = (self.beta2 * self.sdW1) + (1 - self.beta2) * grads['dW1']**2 
        self.sdW2 = (self.beta2 * self.sdW2) + (1 - self.beta2) * grads['dW2']**2
        # self.sdW3 = (self.beta2 * self.sdW3) + (1 - self.beta2) * grads['dW3']**2
        # self.sdW4 = (self.beta2 * self.sdW4) + (1 - self.beta2) * grads['dW4']**2
        # self.sdW5 = (self.beta2 * self.sdW5) + (1 - self.beta2) * grads['dW5']**2
                
        self.sdb1 = (self.beta2 * self.sdb1) + (1 - self.beta2) * grads['db1']**2  
        self.sdb2 = (self.beta2 * self.sdb2) + (1 - self.beta2) * grads['db2']**2 
        # self.sdb3 = (self.beta2 * self.sdb3) + (1 - self.beta2) * grads['db3']**2 
        # self.sdb4 = (self.beta2 * self.sdb4) + (1 - self.beta2) * grads['db4']**2 
        # self.sdb5 = (self.beta2 * self.sdb5) + (1 - self.beta2) * grads['db5']**2 
        
        #Update parameters.
        self.params['W1'] = self.params['W1'] - self.lr * self.vdW1 / (np.sqrt(self.sdW1) + self.epsilon)  
        self.params['W2'] = self.params['W2'] - self.lr * self.vdW2 / (np.sqrt(self.sdW2) + self.epsilon)  
        # self.params['W3'] = self.params['W3'] - self.lr * self.vdW3 / (np.sqrt(self.sdW3) + self.epsilon)  
        # self.params['W4'] = self.params['W4'] - self.lr * self.vdW4 / (np.sqrt(self.sdW4) + self.epsilon)  
        # self.params['W5'] = self.params['W5'] - self.lr * self.vdW5 / (np.sqrt(self.sdW5) + self.epsilon)  

        self.params['b1'] = self.params['b1'] - self.lr * self.vdb1 / (np.sqrt(self.sdb1) + self.epsilon)  
        self.params['b2'] = self.params['b2'] - self.lr * self.vdb2 / (np.sqrt(self.sdb2) + self.epsilon)  
        # self.params['b3'] = self.params['b3'] - self.lr * self.vdb3 / (np.sqrt(self.sdb3) + self.epsilon)  
        # self.params['b4'] = self.params['b4'] - self.lr * self.vdb4 / (np.sqrt(self.sdb4) + self.epsilon)  
        # self.params['b5'] = self.params['b5'] - self.lr * self.vdb5 / (np.sqrt(self.sdb5) + self.epsilon)  

        return self.params

class TanH():
 
    def __init__(self, alpha = 1.7159):
        self.alpha = alpha
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.
            Parameters:
            - X: input tensor.
        """
        self.cache = X
        return self.alpha * np.tanh(X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * (1 - np.tanh(X)**2)
       
class Relu():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def forward(self, X):
        #print("ReLU: _forward")
        self.cache = X
        out = X.copy()
        out[out < 0] = 0
        return out

    def backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = dout.copy()
        dX[X <= 0] = 0
        return dX

class Softmax():
    
    def __init__(self):
        pass

    def forward(self, X):
        """
            Compute softmax values for each sets of scores in X.
            Parameters:
            - X: input vector.
        """
        return np.exp(X) / np.sum(np.exp(X), axis=0)

class CrossEntropyLoss():

    def __init__(self):
        #self.BATCH_SIZE = BATCH_SIZE
        pass
    
    def get(self, y_pred, y):
        """
            Negative log likelihood.
            Parameters:
            -
            -
        """
        return -np.sum(y * np.log(y_pred))



