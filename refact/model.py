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

    def forward(self, X, y):
        conv1 = self.conv1.forward(X) #(28x28x6)
        #print(conv1.shape)

        act1 = self.tanh1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(14x14x6)
        #print(pool1.shape)

        conv2 = self.conv2.forward(pool1) #(10x10x16)
        #print(conv2.shape)
        act2 = self.tanh2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(5x5x16)
        #print(pool2.shape)
        
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(np.prod(pool2.shape[1:]), pool2.shape[0]) #(400x1)
        

        # print(pool2_flatten.shape)
        # print(self.fc1.W['val'].shape)
        # print(self.fc1.b['val'].shape)

        fc1 = self.fc1.forward(pool2_flatten) #(120x1)
        #print(fc1.shape)
        act3 = self.tanh3.forward(fc1)

        fc2 = self.fc2.forward(act3) #(84x1)
        #print(fc2.shape)
        act4 = self.tanh4.forward(fc2)

        fc3 = self.fc3.forward(act4) #(10x1)
        #print(fc3.shape)

        y_pred = self.softmax.forward(fc3)
        #print(y_pred.shape) 

        deltaL = y_pred - y

        return y_pred, deltaL
        
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
        #print(deltaL_4.shape)
       
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
    ITER = 6000
    BATCH_SIZE = 10

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
            y_batch = get_batch(y_train, BATCH_SIZE, t)

            y_pred, deltaL = model.forward(X_batch, y_batch)
            grads = model.backward(deltaL)
            
            params = optimizer.update_params(grads)
            model.set_params(params)
            
            costs.append(cost.get(y_pred, y_batch))
            print(cost.get(y_pred, y_batch))
    
    plt.imshow(costs, cmap='gray')
    plt.show()
    save_params_to_file(model, "weights.pkl")
    

#train(filename)

def predict():
    X_train, y_train, X_test, y_test = load(filename)
    X_train, X_test = X_train/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    model = LeNet5()
    load_params_from_file(model,'weights.pkl')

    y_pred, _ = model.forward(resize_batch(X_test[0]), y_test[0])
    tmp =  np.argmax(y_pred, axis=0)
    print(tmp)

    first_image = X_test[0]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    

#predict()

#--------------------------------------------

class CNN():

    def __init__(self):
        self.conv1 = Conv(nb_filters = 6, filter_size = 2, nb_channels = 1)
        self.tanh1 = TanH()
        self.pool1 = AvgPool(filter_size = 2, stride = 2)
        self.pool1_shape = None
        self.fc1 = Fc(row = 2, column = 24)
        self.softmax = Softmax()

        self.conv1.W['val'] = np.array([[[[ 0.00518085,  0.00728691,  0.05680518, -0.07769635,
                                0.00828898, -0.05789031]],
                                [[ 0.05832487,  0.08991468,  0.02812871, -0.12276262,
                                0.03121113,  0.06930596]]],


                            [[[ 0.00722645,  0.01901338,  0.00163168,  0.08552917,
                                0.08895655,  0.06886178]],

                                [[ 0.02609058, -0.07092969, -0.05189414,  0.04462116,
                                -0.02181925,  0.05610465]]]])

        self.conv1.W['val'] = self.conv1.W['val'].reshape((6,2,2,1))

        self.conv1.b['val'] = np.array([ 0.07422396, -0.03519059,  0.06985363,  0.11874662, -0.01429548,
        0.09669616])
        self.conv1.b['val'] = self.conv1.b['val'].reshape(6,1,1,1)


        self.fc1.W['val'] = np.array([[ 0.37178135, -0.20812127],
                                    [ 0.2213248 ,  0.24179077],
                                    [ 0.14447963,  0.34838623],
                                    [-0.12575299,  0.39683098],
                                    [-0.4641032 , -0.1724233 ],
                                    [-0.402823  ,  0.01583996],
                                    [-0.29589424, -0.05536667],
                                    [ 0.37703496, -0.30489594],
                                    [-0.40505937,  0.19590372],
                                    [-0.25535178,  0.1678713 ],
                                    [-0.20998645,  0.1781227 ],
                                    [ 0.16208494,  0.14354974],
                                    [ 0.01887119,  0.24200541],
                                    [ 0.4118415 ,  0.2604692 ],
                                    [ 0.13190204,  0.38246816],
                                    [-0.4092555 , -0.07783279],
                                    [ 0.42964643,  0.31570607],
                                    [-0.3387671 ,  0.45398277],
                                    [ 0.22038484, -0.4589631 ],
                                    [ 0.281231  , -0.07679352],
                                    [ 0.10532224, -0.24451596],
                                    [-0.44009042,  0.41875094],
                                    [-0.25584435,  0.254641  ],
                                    [ 0.07489949,  0.20691425]])
        self.fc1.W['val'] = self.fc1.W['val'].reshape(2, 24)
        
        self.fc1.b['val'] = np.array([-0.04171324, -0.0364658 ])
        self.fc1.b['val'] = self.fc1.b['val'].reshape(2,1)
        

    def forward(self, X, y):
        
        prettyPrint3D(X)
        print('-----------------------------------------')
        conv1 = self.conv1.forward(X)
        #print(conv1.shape)
        prettyPrint3D(conv1)

        act1 = self.tanh1.forward(conv1)
        pool1 = self.pool1.forward(act1) 
        #print(pool1.shape)

        self.pool1_shape = pool1.shape 
        pool1_flatten = pool1.reshape(np.prod(pool1.shape[1:]), pool1.shape[0])
        #print(pool1_flatten.shape)
        #print(self.fc1.W['val'].shape)

        fc1 = self.fc1.forward(pool1_flatten)
        #print(fc1.shape)
    
        y_pred = self.softmax.forward(fc1)
        #print(y_pred.shape) 

        deltaL = y_pred - y

        return y_pred, deltaL


    def backward(self, deltaL):
        #Compute gradient for weight/bias between fc1 and pool2 and compute 
        #error too (don't need to backpropagate through tanh here).
        deltaL, dW2, db2 = self.fc1.backward(deltaL) #(400x1)
       
        deltaL = deltaL.reshape(self.pool1_shape) #(5x5x16)
        #print(deltaL.shape)

        #Distribute error through pool2 to conv2.
        deltaL = self.pool1.backward(deltaL) #(10x10x16)
        #Distribute error through tanh.
        deltaL = self.tanh1.backward(deltaL)
        #print(deltaL_4.shape)
        
        #Compute gradient for weight/bias at conv2 layer and backpropagate
        #error to conv1 layer.
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

X_train = np.array([[[[ 0.27904129],
         [ 1.01051528],
         [-0.58087813],
         [-0.52516981],
         [-0.57138017]],

        [[-0.92408284],
         [-2.61254901],
         [ 0.95036968],
         [ 0.81644508],
         [-1.523876  ]],

        [[-0.42804606],
         [-0.74240684],
         [-0.7033438 ],
         [-2.13962066],
         [-0.62947496]],

        [[ 0.59772047],
         [ 2.55948803],
         [ 0.39423302],
         [ 0.12221917],
         [-0.51543566]],

        [[-0.60025385],
         [ 0.94743982],
         [ 0.291034  ],
         [-0.63555974],
         [-1.02155219]]],


       [[[-0.16175539],
         [-0.5336488 ],
         [-0.00552786],
         [-0.22945045],
         [ 0.38934891]],

        [[-1.26511911],
         [ 1.09199226],
         [ 2.77831304],
         [ 1.19363972],
         [ 0.21863832]],

        [[ 0.88176104],
         [-1.00908534],
         [-1.58329421],
         [ 0.77370042],
         [-0.53814166]],

        [[-1.3466781 ],
         [-0.88059127],
         [-1.1305523 ],
         [ 0.13442888],
         [ 0.58212279]],

        [[ 0.88774846],
         [ 0.89433233],
         [ 0.7549978 ],
         [-0.20716589],
         [-0.62347739]]]])
y_train = np.array([[1., 0.],[0., 1.]])

params = model.get_params()

# print("W1")
# prettyPrint3D(params['W1'])
# print("b1"), 
# prettyPrint3D(params['b1'])
# print("W2", params['W2'], end = '\n\n')
# print("b2", params['b2'], end = '\n\n')

y_pred, deltaL = model.forward(X_train, y_train)

grads = model.backward(deltaL)
a = optimizer.update_params(grads)
model.set_params(a)

print("--------------------------------------------------")
params = model.get_params()

# print("W1")
# prettyPrint3D(params['W1'])
# print("b1"), 
# prettyPrint3D(params['b1'])
# print("W2", params['W2'], end = '\n\n')
# print("b2", params['b2'], end = '\n\n')
