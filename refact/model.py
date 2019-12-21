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

    def forward(self, X, y):
        conv1 = self.conv1.forward(X) #(28x28x6)
        #print(conv1.shape)

        act1 = self.tanH1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(14x14x6)
        #print(pool1.shape)

        conv2 = self.conv2.forward(pool1) #(10x10x16)
        #print(conv2.shape)
        act2 = self.tanH2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(5x5x16)
        #print(pool2.shape)
        
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(np.prod(pool2.shape[1:]), pool2.shape[0]) #(400x1)
        #print(pool2_flatten.shape)
        
        fc1 = self.fc1.forward(pool2_flatten) #(120x1)
        #print(fc1.shape)
        act3 = self.tanH3.forward(fc1)

        fc2 = self.fc2.forward(act3) #(84x1)
        #print(fc2.shape)
        act4 = self.tanH4.forward(fc2)

        fc3 = self.fc3.forward(act4) #(10x1)
        #print(fc3.shape)

        y_pred = self.softmax.forward(fc3)
        #print(y_pred.shape) 

        deltaL = y_pred - y

        return y_pred, deltaL
        
    def backward(self, deltaL):
        
        #Compute gradient for weight/bias between fc3 and fc2.
        deltaL_1, dW5, db5, = self.fc3.backward(deltaL)
        #Compute error at fc2 layer.
        deltaL_1 = self.tanH4.backward(deltaL_1) #(84x1) 
        #print(deltaL_1.shape)

        #Compute gradient for weight/bias between fc2 and fc1.
        deltaL_2, dW4, db4 = self.fc2.backward(deltaL_1)
        #Compute error at fc1 layer.
        deltaL_2 = self.tanH3.backward(deltaL_2) #(120x1)
        #print(deltaL_2.shape)

        #Compute gradient for weight/bias between fc1 and pool2 and compute 
        #error too (don't need to backpropagate through tanH here).
        deltaL_3, dW3, db3 = self.fc1.backward(deltaL_2) #(400x1)
        deltaL_3 = deltaL_3.reshape(self.pool2_shape) #(5x5x16)
        #print(deltaL_3.shape)

        #Distribute error through pool2 to conv2.
        deltaL_4 = self.pool2.backward(deltaL_3) #(10x10x16)
        #Distribute error through tanH.
        deltaL_4 = self.tanH2.backward(deltaL_4)
        #print(deltaL_4.shape)
       
        #Compute gradient for weight/bias at conv2 layer and backpropagate
        #error to conv1 layer.
        deltaL_4, dW2, db2 = self.conv2.backward(deltaL_4) #(14x14x16)

        #Distribute error through pool1 by creating a temporary pooling layer
        #of conv1 shape.
        deltaL_5 = self.pool1.backward(deltaL_4) #(28x28x6)
        #Distribute error through tanH.
        deltaL_5 = self.tanH1.backward(deltaL_5)
        #Compute gradient for weight/bias at conv1 layer and backpropagate
        #error at conv1 layer.
        deltaL_5, dW1, db1 = self.conv1.backward(deltaL_5)
        #print(deltaL_5.shape)
        
        grads = {
                'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 
                'dW3': dW3, 'db3': db3, 'dW4': dW4, 'db4': db4,
                'dW5': dW5, 'db5': db5
            }

        return grads


    def get_params(self):
        """

        """
        params = {      
            'W1': self.conv1.W['val'], 'b1': self.conv1.b['val'],
            'W2': self.conv2.W['val'], 'b2': self.conv2.b['val'],
            'W3': self.fc1.W['val'], 'b3': self.fc1.b['val'], 
            'W4': self.fc2.W['val'], 'b4': self.fc2.b['val'], 
            'W5': self.fc3.W['val'], 'b5': self.fc3.b['val']
        }  
        return params


filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def get_batch(X, batch_size):
    N = len(X)
    i = np.random.randint(1, N-batch_size)
    return X[i:i+batch_size]


def train(filename):
    X_train, y_train, X_test, y_test = load(filename)
    X_train, X_test = X_train/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    model = LeNet5()
    cost = CrossEntropyLoss()
    optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    
    
    costs = []
    NB_EPOCH = 10
    BATCH_SIZE = 32
    
    for epoch in range(NB_EPOCH):

        X_batch = resize_batch(get_batch(X_train, BATCH_SIZE))
        y_batch = get_batch(y_train, BATCH_SIZE)

        y_pred, deltaL = model.forward(X_batch, y_batch)
        grads = model.backward(deltaL)
        optimizer.update_params(grads)
        #costs.append(cost.get(y_pred, y_resize))
        print(cost.get(y_pred, y_batch))
    
    """
    y_pred, deltaL = model.forward(resize_batch(X_train[0]), y_train[0])
    grads = model.backward(deltaL)
    print('Done')
    """


train(filename)