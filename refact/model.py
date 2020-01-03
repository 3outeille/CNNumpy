#-----
#Model
#-----
#import numpy as np

from new_style import *
from utils import *
import numpy as np
import pickle
import matplotlib.pyplot as plt


class LeNet5():

    def __init__(self):
        self.conv1 = Conv(nb_filters = 6, filter_size = 5, nb_channels = 1)
        self.tanh1 = TanH()
        self.pool1 = AvgPool(filter_size = 2, stride = 2)
        self.conv2 = Conv(nb_filters = 16, filter_size = 5, nb_channels = 6)
        self.tanh2 = TanH()
        self.pool2 = AvgPool(filter_size = 2, stride = 2)
        self.pool2_shape = None
        self.fc1 = Fc(row = 120, column = 5*5*16)
        self.tanh3 = TanH()
        self.fc2 = Fc(row = 84, column = 120)
        self.tanh4 = TanH()
        self.fc3 = Fc(row = 10 , column = 84)
        self.softmax = Softmax()

    def forward(self, X):
        conv1 = self.conv1.forward(X) #(6x28x28)
        # print(conv1.shape)

        act1 = self.tanh1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(6x14x14)
        # print(pool1.shape)

        conv2 = self.conv2.forward(pool1) #(16x10x10)
        # print(conv2.shape)
        act2 = self.tanh2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(16x5x5)
        # print(pool2.shape)
        
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(np.prod(pool2.shape[1:]), pool2.shape[0]) #(400x1)
        
        fc1 = self.fc1.forward(pool2_flatten) #(120x1)
        # print(fc1.shape)
        act3 = self.tanh3.forward(fc1)

        fc2 = self.fc2.forward(act3) #(84x1)
        # print(fc2.shape)
        act4 = self.tanh4.forward(fc2)

        fc3 = self.fc3.forward(act4) #(10x1)
        # print(fc3.shape)

        y_pred = self.softmax.forward(fc3)
        # print(y_pred.shape) 

        return y_pred
        
    def backward(self, deltaL):
        #Compute gradient for weight/bias between fc3 and fc2.
        deltaL, dW5, db5, = self.fc3.backward(deltaL)
        #Compute error at fc2 layer.
        deltaL = self.tanh4.backward(deltaL) #(84x1) 
        #print(deltaL_1.shape)

        #Compute gradient for weight/bias between fc2 and fc1.
        deltaL, dW4, db4 = self.fc2.backward(deltaL)
        #Compute error at fc1 layer.
        deltaL = self.tanh3.backward(deltaL) #(120x1)
        #print(deltaL_2.shape)

        #Compute gradient for weight/bias between fc1 and pool2 and compute 
        #error too (don't need to backpropagate through tanh here).
        deltaL, dW3, db3 = self.fc1.backward(deltaL) #(400x1)
        deltaL = deltaL.reshape(self.pool2_shape) #(5x5x16)
        #print(deltaL_3.shape)

        #Distribute error through pool2 to conv2.
        deltaL = self.pool2.backward(deltaL) #(10x10x16)
        #Distribute error through tanh.
        deltaL = self.tanh2.backward(deltaL)
       
        #Compute gradient for weight/bias at conv2 layer and backpropagate
        #error to conv1 layer.
        deltaL, dW2, db2 = self.conv2.backward(deltaL) #(14x14x16)

        #Distribute error through pool1 by creating a temporary pooling layer
        #of conv1 shape.
        deltaL = self.pool1.backward(deltaL) #(28x28x6)
        #Distribute error through tanh.
        deltaL = self.tanh1.backward(deltaL)
        #Compute gradient for weight/bias at conv1 layer and backpropagate
        #error at conv1 layer.
        deltaL, dW1, db1 = self.conv1.backward(deltaL)
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

    def set_params(self, params):
        """

        """
        self.conv1.W['val'] = params['W1']
        self.conv2.W['val'] = params['W2']
        self.fc1.W['val'] = params['W3']
        self.fc2.W['val'] = params['W4']
        self.fc3.W['val'] = params['W5'] 

        self.conv1.b['val'] = params['b1']
        self.conv2.b['val'] = params['b2']
        self.fc1.b['val'] = params['b3']
        self.fc2.b['val'] = params['b4']
        self.fc3.b['val'] = params['b5'] 

        

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def train(filename):
    NB_EPOCH = 1
    ITER = 938
    BATCH_SIZE = 2

    X_train, y_train, X_test, y_test = load(filename)
    X_train, X_test = X_train/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    model = LeNet5()
    cost = CrossEntropyLoss()
    optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    
    
    costs = []

    for epoch in range(NB_EPOCH):

        for t in range(ITER):
            X_batch = resize_batch(get_batch(X_train, BATCH_SIZE, t))
            y_batch_encoded = one_hot_encoding(get_batch(y_train, BATCH_SIZE, t))

            y_pred = model.forward(X_batch)
            loss, deltaL = cost.get(y_pred, y_batch_encoded)
            grads = model.backward(deltaL)
            params = optimizer.update_params(grads)
            model.set_params(params)

            """
            IDEA: transform y_pred = [0, 0.45, ..., 0,9] -> [0, 0, ..., 1]
                +
            Check (10, batch_size) or (batc_size, 10) of measure_performance()
            """

            #Not working !
            # print(y_pred)
            # tmp = np.zeros((10, BATCH_SIZE))
            # tmp[np.arange(BATCH_SIZE), np.argmax(y_pred, axis = 0)] = 1

            costs.append(loss)
            accuracy = measure_performance(y_pred, y_batch_encoded)

            print('[Epoch {} | ITER {}] Loss: {} | Accuracy: {}'.format(epoch+1, t+1, loss, accuracy))

            if loss < 3:
                save_params_to_file(model, "loss_inferior_2.pkl")

    plt.plot(costs)
    plt.show()
    save_params_to_file(model, "weights.pkl")

train(filename)

def predict():
    X_train, y_train, X_test, y_test = load(filename)
    X_train, X_test = X_train/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    model = LeNet5()
    load_params_from_file(model,'weights.pkl')
    
    image = resize_batch(X_train[0])
    y_pred = model.forward(image)
    tmp = np.argmax(y_pred, axis=0)
    print(tmp)

    first_image = image
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((32, 32))
    plt.imshow(pixels, cmap='gray')
    plt.show()

predict()

#--------------------------------------------
"""
class CNN():

    def __init__(self):
        self.conv1 = Conv(nb_filters = 6, filter_size = 2, nb_channels = 1)
        self.tanh1 = TanH()
        self.pool1 = AvgPool(filter_size = 2, stride = 2)
        self.pool1_shape = None
        self.fc1 = Fc(row = 2, column = 24)
        self.softmax = Softmax()


    def forward(self, X, y):    
        conv1 = self.conv1.forward(X)
        act1 = self.tanh1.forward(conv1)
        pool1 = self.pool1.forward(act1) 
        self.pool1_shape = pool1.shape 
        pool1_flatten = pool1.reshape(np.prod(pool1.shape[1:]), pool1.shape[0])
        fc1 = self.fc1.forward(pool1_flatten)
        y_pred = self.softmax.forward(fc1)
        deltaL = y_pred - y

        return y_pred, deltaL


    def backward(self, deltaL):
        deltaL, dW2, db2 = self.fc1.backward(deltaL)
        deltaL = deltaL.reshape(self.pool1_shape)
        deltaL = self.pool1.backward(deltaL)
        deltaL = self.tanh1.backward(deltaL)
        deltaL, dW1, db1 = self.conv1.backward(deltaL) #(14x14x16)
        
        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }

        return grads

    def get_params(self):
        
        params = {      
            'W1': self.conv1.W['val'], 'b1': self.conv1.b['val'],
            'W2': self.fc1.W['val'], 'b2': self.fc1.b['val']
        }
        return params

    def set_params(self, a):
        self.conv1.W['val'] = a['W1']
        self.conv1.b['val'] = a['b1']

        self.fc1.W['val'] = a['W2']
        self.fc1.b['val'] = a['b2']


model = CNN()
cost = CrossEntropyLoss()
optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    

X_train = np.array([[[[ 0.49671415],
         [-0.1382643 ],
         [ 0.64768854],
         [ 1.52302986],
         [-0.23415337]],

        [[-0.23413696],
         [ 1.57921282],
         [ 0.76743473],
         [-0.46947439],
         [ 0.54256004]],

        [[-0.46341769],
         [-0.46572975],
         [ 0.24196227],
         [-1.91328024],
         [-1.72491783]],

        [[-0.56228753],
         [-1.01283112],
         [ 0.31424733],
         [-0.90802408],
         [-1.4123037 ]],

        [[ 1.46564877],
         [-0.2257763 ],
         [ 0.0675282 ],
         [-1.42474819],
         [-0.54438272]]],


       [[[ 0.11092259],
         [-1.15099358],
         [ 0.37569802],
         [-0.60063869],
         [-0.29169375]],

        [[-0.60170661],
         [ 1.85227818],
         [-0.01349722],
         [-1.05771093],
         [ 0.82254491]],

        [[-1.22084365],
         [ 0.2088636 ],
         [-1.95967012],
         [-1.32818605],
         [ 0.19686124]],

        [[ 0.73846658],
         [ 0.17136828],
         [-0.11564828],
         [-0.3011037 ],
         [-1.47852199]],

        [[-0.71984421],
         [-0.46063877],
         [ 1.05712223],
         [ 0.34361829],
         [-1.76304016]]]])
X_train = X_train.reshape(2, 1, 5, 5)
y_train = np.array([[1., 0.],[0., 1.]])

# params = model.get_params()
# print("W1")
# prettyPrint3D(params['W1'])
# print("W2", params['W2'], end = '\n\n')

y_pred, deltaL = model.forward(X_train, y_train)
grads = model.backward(deltaL)
a = optimizer.update_params(grads)
model.set_params(a)

# params = model.get_params()
# print("W1")
# prettyPrint3D(params['W1'])
# print("W2", params['W2'], end = '\n\n')
"""