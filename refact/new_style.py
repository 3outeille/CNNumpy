import numpy as np
import urllib.request
import gzip
import os
from skimage import transform


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
        
        self.cache = None

    def forward(self, A_prev):
        """
            Performs a forward convolution.
            Parameters:
            - A_prev: Last layer of shape (m, n_H_prev, n_W_prev, n_C_prev).
                     If first convolution, A_prev = X.
            Returns:
            - A_conv: previous layer convolved.
        """
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        n_C = self.n_F

        A_conv = np.zeros((m, n_H, n_W, n_C))
        
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
                         
                        A_conv[i, h, w, c] = np.sum(A_prev[i, h_start:h_end, w_start:w_end] 
                                        * self.W['val'][c, ...] + self.b['val'][c, ...])
        self.cache = A_conv

        return A_conv 

    def backward(self, A_prev_error):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - A_prev_error: error from previous layer.
            Returns:
            - deltaL: error of the current convolutional layer.
        """
        
        W_rot = np.rot90(np.rot90(self.W['val']))
        
        X = self.cache
        m, n_H, n_W, n_C = X.shape
        deltaL = np.zeros(X.shape)
    
        #Distribute error in weight (compute gradient of weight).
        for i in range(self.n_F):
            for h in range(self.f):
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(self.f):
                    w_start = w * self.s
                    w_end = w_start + self.f

                    self.W['grad'][i, h, w] = np.sum(X[i, h_start:h_end, w_start:w_end] 
                                            * A_prev_error[i, h, w])

        #Distribute error in bias (compute gradient of bias).
        for i in range(self.n_F):
            self.b['grad'][i] = np.sum(A_prev_error[i])

        A_prev_error_pad = np.pad(A_prev_error, ((0, 0), (self.f, self.f), 
                        (self.f, self.f), (0, 0)), 'constant')

        #Compute error of current layer.
        for i in range(m):
            for h in range(n_H):
                h_start = h * self.s
                h_end = h_start + self.f
                
                for w in range(n_W): #Slide the filter horizontally.                
                    w_start = w * self.s
                    w_end = w_start + self.f
                    
                    for c in range(n_C):
                        deltaL[i, h, w, c] = np.sum(W_rot * A_prev_error_pad[i, h_start:h_end, w_start:w_end, c]) 

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
            - A_conv_act: Output of activation function.
            Returns:
            - A_pool: A_conv_act squashed. 
        """
        m, n_H_prev, n_W_prev, n_C_prev = A_conv_act.shape
        
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
                    
                    A_pool[i, h, w] = np.mean(A_conv_act[i, h_start:h_end, w_start:w_end])
        
        self.cache = A_pool

        return A_pool

    def backward(self, A_prev_error):
        """
            Distributes the error back from previous layer to pooling layer.
            Parameters:
            - A_prev_error: Previous layer with the error.
            Returns:
            - A_pool_new: New pooling layer updated with error.

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
                        
                        A_pool_new[i, h_start:h_end, w_start:w_end, c] +=  filter_average

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


#-----
#Model
#-----

class LeNet5():

    def __init__(self):
        self.conv1 = Conv(nb_filters = 6, filter_size = 5, nb_channels = 1)
        self.tanH1 = TanH()
        self.pool1 = AvgPool(filter_size = 2, stride = 2)

        self.conv2 = Conv(nb_filters = 16, filter_size = 5, nb_channels = 6)
        self.tanH2 = TanH()
        self.pool2 = AvgPool(filter_size = 2, stride = 2)
        self.pool2_shape = None

        self.fc1 = Fc(row = 120, column = 5*5*16)
        self.tanH3 = TanH()

        self.fc2 = Fc(row = 84, column = 120)
        self.tanH4= TanH()

        self.fc3 = Fc(row = 10 , column = 84)

        self.softmax = Softmax()

    def forward(self, X):
        conv1 = self.conv1.forward(X) #(28 x 28 x 6)
        print(conv1.shape)

        act1 = self.tanH1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(14 x 14 x 6)
        print(pool1.shape)

        conv2 = self.conv2.forward(pool1) #(10 x 10 x 16)
        print(conv2.shape)
        act2 = self.tanH2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(5 x 5 x 16)
        print(pool2.shape)
        
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(np.prod(pool2.shape), 1) #(400 x 1)
        print(pool2_flatten.shape)

        fc1 = self.fc1.forward(pool2_flatten) #(120 x 1)
        print(fc1.shape)
        act3 = self.tanH3.forward(fc1)

        fc2 = self.fc2.forward(act3) #(84 x 1)
        print(fc2.shape)
        act4 = self.tanH4.forward(fc2)

        fc3 = self.fc3.forward(act4) #(10 x 1)
        print(fc3.shape)

        yHat = self.softmax.forward(fc3)
        print(yHat.shape) 

        return yHat
        
    def backward(self, deltaL):
        #We suppose that deltaL = yHat - y (10 x 1)
        
        #Compute gradient for weight/bias between fc3 and fc2.
        grad1 = self.fc3.backward(deltaL)
        #Compute error at fc2 layer.
        deltaL_1 = self.tanH4.backward(grad1) #(84 x 1) 
        print(deltaL_1.shape)

        #Compute gradient for weight/bias between fc2 and fc1.
        grad2 = self.fc2.backward(deltaL_1)
        #Compute error at fc1 layer.
        deltaL_2 = self.tanH3.backward(grad2) #(120 x 1)
        print(deltaL_2.shape)

        #Compute gradient for weight/bias between fc1 and pool2
        #and the error too (don't need to backpropagate through tanH).
        deltaL_3 = self.fc1.backward(deltaL_2) #(400 x 1)
        deltaL_3 = deltaL_3.reshape(self.pool2_shape) #(5 x 5 x 16)
        print(deltaL_3.shape)      

        #Distribute error in pool2.
        deltaL_4 = self.pool2.backward(deltaL_3) #(5 x 5 x 16)
        #Distribute error through tanH.
        deltaL_4 = self.tanH2.backward(deltaL_4)
        #Distribute error from pool2 to conv2.
        deltaL_4 = self.conv2.backward(deltaL_4) #(10 x 10 x 16)
        print(deltaL_4.shape)

        #Distribute error to pool1.
        deltaL_5 = self.pool1.backward(deltaL_4) #(14 x 14 x 6)
        #Distribute error through tanH.
        deltaL_5 = self.tanH1.backward(deltaL_5)
        #Distribute error from pool1 to conv1.
        deltaL_5 = self.conv1.backward(deltaL_5) #(28 x 28 x 6)
        print(deltaL_5.shape)
        
    
    def get_params(self):
        pass

    def set_params(self):
        pass


#------------------
#Utilities function
#------------------


filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist(filename):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for elt in filename:
        print("Downloading " + elt[1] + " in data/ ...")
        urllib.request.urlretrieve(base_url+elt[1], 'data/' + elt[1])
    print("Download complete.")


def extract_mnist(filename):
    mnist = {}
    for elt in filename[:2]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open('data/' + elt[1]) as f:
            #According to the doc on MNIST website, offset for image starts at 16.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28*28)
    
    for elt in filename[2:]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open('data/' + elt[1]) as f:
            #According to the doc on MNIST website, offset for label starts at 8.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    print('Extraction complete') 

    return mnist

def load(filename):
    L = [elt[1] for elt in filename]   
    count = 0 

    #Check if the 4 .gz files exist.
    for elt in L:
        if os.path.isfile('data/' + elt):
            count += 1

    #If the 4 .gz are not in data/, we download and extract them.
    if count != 4:
        download_mnist(filename)
        mnist = extract_mnist(filename)
    else: #We just extract them.
        mnist = extract_mnist(filename)

    print('Loading complete')
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
        

def resize_batch(imgs):
    """
        A function to resize a batch of MNIST images to (32, 32)

        Parameters:
        -imgs: a numpy array of size [batch_size, 28 X 28].
    """
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

def train(filename):
    X_train, Y_train, X_test, Y_test = load(filename)
    X_train, X_test = X_train/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    
    X_resize = resize_batch(X_train[0])
    print(X_resize.shape)

    model = LeNet5()
    Y_pred = model.forward(X_resize)
    deltaL = Y_pred - Y_train[0]
    print('-----------')
    model.backward(deltaL)

train(filename)
