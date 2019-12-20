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
                
                W_rot_tile = np.repeat(W_rot[i, :, :, c][..., np.newaxis], repeats = n_C_dout , axis = 2)
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

        return dX
        
        
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
            Distributes the error back from previous layer to pooling layer.
            Parameters:
            - A_prev_error: Previous layer with the error.
            Returns:
            - A_pool_new: New pooling layer updated with error.

        """
        """
        A_pool = self.cache
        m, n_H, n_W, n_C = A_pool.shape
        A_pool_new = np.copy(A_prev_error)
        
        for i in range(m):

            for h in range(n_H):
                h_start = h * self.s
                h_end = h_start + self.f

                for w in range(n_W):
                    w_start = w * self.s
                    w_end = w_start + self.f

                    for c in range(n_C):
                        #Compute average for a given value.
                        average = A_pool[i, h, w, c] / (n_H * n_W)
                        #Create a filter of size (f, f) filled with average.
                        filter_average = np.full((self.f, self.f) , average)
                        
                        print('filter_average:' + str(filter_average.shape))
                        print('A_pool_new:' + str(A_pool_new[i, h_start:h_end, w_start:w_end, c].shape)) 

                        A_pool_new[i, h_start:h_end, w_start:w_end, c] +=  filter_average

        return A_pool_new
        """
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
        A_prev = self.cache
        m = A_prev.shape[0]
        
        #Compute gradient.
        self.W['grad'] = (1/m) * np.dot(deltaL, A_prev.T)
        self.b['grad'] = (1/m) * np.sum(deltaL, axis = 1)
        
        #Compute error.
        new_deltaL = np.dot(self.W['val'].T, deltaL) 
        #We still need to multiply new_deltaL by the derivative of the activation
        #function which is done in TanH.backward() only when we have to backpropagate
        #error through tanH.

        return new_deltaL
    
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
            - new_deltaL: error computed in Fc.backward().
        """
        X = self.cache
        return new_deltaL * (1 - np.tanh(X)**2)

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