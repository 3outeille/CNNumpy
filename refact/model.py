#-----
#Model
#-----
#import numpy as np

from new_style import *
from utils import *
import numpy as np

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
        conv1 = self.conv1.forward(X) #(28x28x6)
        print(conv1.shape)

        act1 = self.tanH1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(14x14x6)
        print(pool1.shape)

        conv2 = self.conv2.forward(pool1) #(10x10x16)
        print(conv2.shape)
        act2 = self.tanH2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(5x5x16)
        print(pool2.shape)
        
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(np.prod(pool2.shape), 1) #(400x1)
        print(pool2_flatten.shape)

        fc1 = self.fc1.forward(pool2_flatten) #(120x1)
        print(fc1.shape)
        act3 = self.tanH3.forward(fc1)

        fc2 = self.fc2.forward(act3) #(84x1)
        print(fc2.shape)
        act4 = self.tanH4.forward(fc2)

        fc3 = self.fc3.forward(act4) #(10x1)
        print(fc3.shape)

        yHat = self.softmax.forward(fc3)
        print(yHat.shape) 

        return yHat
        
    def backward(self, deltaL):
        #We suppose that deltaL = yHat - y (10x1)
        
        #Compute gradient for weight/bias between fc3 and fc2.
        grad1 = self.fc3.backward(deltaL)
        #Compute error at fc2 layer.
        deltaL_1 = self.tanH4.backward(grad1) #(84x1) 
        print(deltaL_1.shape)

        #Compute gradient for weight/bias between fc2 and fc1.
        grad2 = self.fc2.backward(deltaL_1)
        #Compute error at fc1 layer.
        deltaL_2 = self.tanH3.backward(grad2) #(120x1)
        print(deltaL_2.shape)

        #Compute gradient for weight/bias between fc1 and pool2 and compute 
        #error too (don't need to backpropagate through tanH here).
        deltaL_3 = self.fc1.backward(deltaL_2) #(400x1)
        deltaL_3 = deltaL_3.reshape(self.pool2_shape) #(5x5x16)
        print(deltaL_3.shape)

        #Distribute error through pool2 to conv2.
        deltaL_4 = self.pool2.backward(deltaL_3) #(10x10x16)
        #Distribute error through tanH.
        deltaL_4 = self.tanH2.backward(deltaL_4)
        print(deltaL_4.shape)
       
        #Compute gradient for weight/bias at conv2 layer and backpropagate
        #error to conv1 layer.
        deltaL_4 = self.conv2.backward(deltaL_4) #(14x14x16)

        #Distribute error through pool1 by creating a temporary pooling layer
        #of conv1 shape.
        deltaL_5 = self.pool1.backward(deltaL_4) #(28x28x6)
        #Distribute error through tanH.
        deltaL_5 = self.tanH1.backward(deltaL_5)
        #Compute gradient for weight/bias at conv1 layer and backpropagate
        #error at conv1 layer.
        deltaL_5 = self.conv1.backward(deltaL_5)
        print(deltaL_5.shape)

    
    def get_params(self):
        pass

    def set_params(self):
        pass

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

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