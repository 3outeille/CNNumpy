import numpy as np
import math

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


        #Xavier initialization.
        bound = 1 / math.sqrt(self.f * self.f)
        self.W = {'val': np.random.uniform(-bound, bound, size=(self.n_F, self.n_C, self.f, self.f)),
                  'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
     
        self.b = {'val': np.random.uniform(-bound, bound, size=(self.n_F)),
                  'grad': np.zeros((self.n_F))}

        self.cache = None

    def forward(self, X):
        """
            Performs a forward convolution.
           
            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - A_conv: previous layer convolved.
        """
        self.cache = X
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1

        out = np.zeros((m, n_C, n_H, n_W))

        for i in range(m): #For each image.
            
            for c in range(n_C): #For each channel.

                for h in range(n_H): #Slide the filter vertically.
                    h_start = h * self.s
                    h_end = h_start + self.f
                    
                    for w in range(n_W): #Slide the filter horizontally.                
                        w_start = w * self.s
                        w_end = w_start + self.f

                        out[i, c, h, w] = np.sum(X[i, :, h_start:h_end, w_start:w_end] 
                                        * self.W['val'][c, ...]) + self.b['val'][c]
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
        
        m, n_C, n_H, n_W = X.shape
        m, n_C_dout, n_H_dout, n_W_dout = dout.shape
        
        W_rot = np.rot90(np.rot90(self.W['val']))
        dX = np.zeros(X.shape)

        #Compute dW.
        for i in range(m): #For each examples.
            
            for c in range(n_C_dout):
                
                for h in range(n_H_dout):
                    h_start = h * self.s
                    h_end = h_start + self.f

                    for w in range(n_W_dout):
                        w_start = w * self.s
                        w_end = w_start + self.f

                        self.W['grad'][c, ...] += dout[i, c, h, w] * X[i, :, h_start:h_end, w_start:w_end]
                        dX[i, :, h_start:h_end, w_start:w_end] += dout[i, c, h, w] * W_rot[c, ...]
        #Compute db.
        for c in range(self.n_F):
            self.b['grad'][c, ...] = np.sum(dout[:, c, ...])
        
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
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        
        n_C = n_C_prev
        n_H = int((n_H_prev - self.f)/ self.s) + 1
        n_W = int((n_W_prev - self.f)/ self.s) + 1

        A_pool = np.zeros((m, n_C, n_H, n_W))
    
        for i in range(m): #For each image.
            
            for h in range(n_H): #Slide the filter vertically.
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    A_pool[i, :, h, w] = np.mean(X[i, :, h_start:h_end, w_start:w_end])
        
        self.cache = X

        return A_pool

    def backward(self, dout):
        """
            Distributes error through pooling layer.

            Parameters:
            - dout: Previous layer with the error.
            
            Returns:
            - dX: Conv layer updated with error.
        """
        X = self.cache
        m, n_C, n_H, n_W = dout.shape
        dX = np.copy(X)        

        for i in range(m):
            
            for c in range(n_C):

                for h in range(n_H):
                    h_start = h * self.s
                    h_end = h_start + self.f

                    for w in range(n_W):
                        w_start = w * self.s
                        w_end = w_start + self.f
                    
                        average = dout[i, c, h, w] / (n_H * n_W)
                        filter_average = np.full((self.f, self.f), average)
                        dX[i, c, h_start:h_end, w_start:w_end] += filter_average

        return dX

class Fc():

    def __init__(self, row, column):
        self.row = row
        self.col = column
        
        #Initialize Weight/bias.
        bound = 1 / np.sqrt(self.row)
        self.W = {'val': np.random.uniform(low=-bound, high=bound, size=(self.row, self.col)), 'grad': 0}
        self.b = {'val': np.random.uniform(low=-bound, high=bound, size=(1, self.row)), 'grad': 0}
        
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
        A_fc = np.dot(fc, self.W['val'].T) + self.b['val']
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
    
        self.W['grad'] = (1/m) * np.dot(deltaL.T, fc)
        self.b['grad'] = (1/m) * np.sum(deltaL, axis = 0)

        #Compute error.
        new_deltaL = np.dot(deltaL, self.W['val']) 
        #We still need to multiply new_deltaL by the derivative of the activation
        #function which is done in TanH.backward().

        return new_deltaL, self.W['grad'], self.b['grad']
    
class SGD():

    def __init__(self, lr, params):
        self.lr = lr
        self.params = params

    def update_params(self, grads):
        #print('Update Params')
        for key in self.params:
            self.params[key] += - self.lr * grads['d' + key]

        return self.params        

class AdamGD():

    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params
        
        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):

        for key in self.params:
            # Momentum update.
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['d' + key] 
            # RMSprop update.
            self.rmsprop['sd' + key] =  (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * grads['d' + key]**2 
            # Update parameters.
            self.params[key] += -self.lr * self.momentum['vd' + key] / (np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)  

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

class Softmax():
    
    def __init__(self):
        pass

    def forward(self, X):
        """
            Compute softmax values for each sets of scores in X.

            Parameters:
            - X: input vector.
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1)[:, np.newaxis]

class CrossEntropyLoss():

    def __init__(self):
        pass
    
    def get(self, y_pred, y):
        """
            Return the negative log likelihood and the error at the last layer.
            
            Parameters:
            - y_pred: model predictions.
            - y: true ground labels.
        """
        batch_size = y_pred.shape[1]
        deltaL = y_pred - y
        loss = -np.sum(y * np.log(y_pred)) / batch_size
        return loss, deltaL
        