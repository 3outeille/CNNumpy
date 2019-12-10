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
        self.W = {'val': np.random.randn(self.n_F, self.f, self.f, self.n_C), 'grad': 0}
        self.b = {'val': np.random.randn(self.n_F, 1, 1, self.n_C), 'grad': 0}
        
    def forward(self, A_prev):
        """
            Performs a forward convolution.

            Parameters:
            -A_prev: Last layer of shape (m, n_H_prev, n_W_prev, n_C_prev).
                     If first convolution, A_prev = X.
            Returns:
            -A_conv: previous layer convolved.
        """

        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        n_C = self.n_F

        A_conv = np.zeros((m, n_H, n_W, n_C))

        for i in range(m): #For each image.
            
            for h in range(n_H): #Slide the filter vertically.
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    A_conv[i, h, w] = np.sum(A_prev[i, h_start:h_end, w_start:w_end, c] 
                                    * self.W['val']) + self.b['val']

        return A_conv 

    def backward(self, A_prev_error):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.

            Parameters:
            -A_prev_error: error from previous layer.
            
            Returns:
            -deltaL: error of the current convolutional layer.
        """
        
        W_rot = np.rot90(np.rot90(self.W['val']))
        
        deltaL = 
        
    
        #Distribute error in weight (compute gradient of weight).
        for i in range(self.W['val'].shape(0)):
            for h in range(self.W['val'].shape(1)):
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(self.W['val'].shape(2)):
                    w_start = w * self.s
                    w_end = w_start + self.f

                    self.W['grad']

        #Distribute error in bias (compute gradient of bias).
    

        #Compute error.
        for i in range(m):
            for h in range(n_H):
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    deltaL[i, h, w] = np.sum(W_rot * A_prev_error[i, h_start:h_end, w_start:w_end]) 

        return deltaL 

class AvgPool():
    
    def __init__(self, filter_size, stride):
        self.f = filter_size
        self.s = stride
        self.cache = None

    def forward(self, A_conv_act):
        """
            Apply average pooling on A_conv_act.

            Parameters:
            -A_conv_act: Output of activation function.

            Returns:
            -A_pool: A_conv_act squashed. 
        """
        m, n_H_prev, n_W_prev, n_C_prev = A_conv_act.shape
        
        n_H = int((n_H_prev - self.f)/ self.s) + 1
        n_W = int((n_W_prev - self.f)/ self.s) + 1
        n_C = self.n_F

        A_pool = np.zeros((m, n_H, n_W, n_C))
    
        for i in range(m): #For each image.
            
            for h in range(n_H): #Slide the filter vertically.
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    A_pool[i, h, w] = np.mean(A_conv_act[i, h_start:h_end, w_start:w_end])
        
        self.cache = A_pool

        return A_pool

    def backward(self, A_prev_error):
        """
            Distributes the error back from previous layer to pooling layer.

            Parameters:
            -A_prev_error: Previous layer with the error.

            Returns:
            -A_pool_new: New pooling layer updated with error.

        """
        A_pool = self.cache
        m, n_H, n_W, n_C = A_pool.shape
        A_pool_new = np.copy(A_prev_error)

        for i in range(m):

            for h in range(n_H):
                h_start = h * s
                h_end = h_start + f

                for w in range(n_W):
                    w_start = w * s
                    w_end = w_start + f

                    for c in range(n_C):
                        #Compute average for a given value.
                        average = A_pool[i, h, w, c] / (n_H * n_W)
                        #Create a filter of size (f, f) filled with average.
                        filter_average = np.full((f, f) , average)

                        A_pool_new[i, h_start:h_end, w_start:w_end, c] += filter_average

        return A_pool_new

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
            -fc: fully connected layer.

            Returns:
            -A_fc: new fully connected layer.
        """
        self.cache = fc
        A_fc = np.dot(self.W['val'], fc) + self.b['val']
        return A_fc

    def backward(self, deltaL):
        """
            Returns the error of the current layer and compute gradients.
            
            Parameters:
            -deltaL: error at last layer.

            Returns:
            -new_deltaL: error at current layer.
        """
        A_prev = self.cache
        m = A_prev.shape(0)
        
        #Compute gradient.
        self.W['grad'] = (1/m) * np.dot(deltaL, A_prev.T)
        self.b['grad'] = (1/m) * np.sum(deltaL, axis = 1)
        
        #Compute error.
        new_deltaL = np.dot(self.W['val'].T, deltaL) 
        #We still need to multiply new_deltaL by the derivative of the activation
        #function which is done in TanH.backward() only when we have to backpropagate
        #error through tanH.

        return new_deltaL
    
#------------------
#Utilities function
#------------------

class TanH():
    def __init__(self, alpha = 1.7159):
        self.alpha = alpha
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.

            Parameters:
            -X: input tensor.
        """
        self.cache = X
        return self.alpha * np.tanh(X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            
            Parameters:
            -new_deltaL: error computed in Fc.backward().
        """
        return new_deltaL * (1 - np.tanh(self.cache)**2)

class Softmax():
    
    def __init__(self):
        pass

    def forward(self):
        pass
#-----
#Model
#-----

class LeNet5():

    def __init__(self):
        self.conv1 = Conv(nb_filters = 6, filter_size = 5, nb_channels = 1)
        self.tanH1 = TanH()
        self.pool1 = AvgPool(2,2)

        self.conv2 = Conv(nb_filters = 16, filter_size = 5, nb_chall 6)
        self.tanH2 = TanH()
        self.pool2 = AvgPool(2,2)
        self.pool2_shape = None

        self.fc1 = Fc(row = 120, column = 5*5*16)
        self.tanH3 = TanH()

        self.fc2 = Fc(row = 84, column = 120)
        self.tanH4= TanH()

        self.fc3 = Fc(row = 10 , column = 84)

        self.softmax = Softmax()

    def forward(self, X):
        conv1 = self.conv1.forward(X) #(28 x 28 x 6)
        act1 = self.tanH1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(14 x 14 x 6)

        conv2 = self.conv2.forward(pool1) #(10 x 10 x 6)
        act2 = self.tanH2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(5 x 5 x 16)

        self.pool2_shape = pool2.shape
        pool2_flatten = pool2.flatten() #(400 x 1)

        fc1 = self.fc1.forward(pool2_flatten) #(120 x 1)
        act3 = self.tanH3.forward(fc1)

        fc2 = self.fc2.forward(act3) #(84 x 1)
        act4 = self.tanH4.forward(fc2)

        fc3 = self.yHat.forward(act4) #(10 x 1)

        yHat = self.softmax(yhat)
        
        return yHat
        
    def backward(self, deltaL):
        #We suppose that deltaL = yHat - y (10 x 1)
        
        #Compute gradient for weight/bias between fc3 and fc2.
        grad1 = self.fc3.backward(deltaL)
        #Compute error at fc2 layer.
        deltaL_1 = self.tanH4.backward(grad1) #(84 x 1) 
        
        #Compute gradient for weight/bias between fc2 and fc1.
        grad2 = self.fc2.backward(deltaL_1)
        #Compute error at fc1 layer.
        deltaL_2 = self.tanH3.backward(grad2) #(120 x 1)
        
        #Compute gradient for weight/bias between fc1 and pool2
        #and the error too (don't need to backpropagate through tanH).
        deltaL_3 = self.fc1.backward(deltaL_2) #(400 x 1)
        deltaL_3 = deltaL_3.reshape(self.pool2_shape) #(5 x 5 x 16)
        
        #Distribute error to pool2.
        deltaL_4 = self.pool2.backward(deltaL_3) #(5 x 5 x 16)
        #Distribute error through tanH.
        deltaL_4 = self.tanH2.backward(deltaL_4)
        #Distribute error from pool2 to conv2.
        deltaL_4 = self.conv2.backward(deltaL_4) #(10 x 10 x 6)
        
        #Distribute error to pool1.
        deltaL_5 = self.pool1.backward(deltaL_4) #(14 x 14 x 6)
        #Distribute error through tanH.
        deltaL_5 = self.tanH1.backward(deltaL_5)
        #Distribute error from pool1 to conv1.
        deltaL_5 = self.conv1.backward(deltaL_5) #(28 x 28 x 6)

    def get_params(self):
        pass

    def set_params(self):
        pass

