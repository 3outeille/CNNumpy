from layers import *
from utils import *
import numpy as np

class NN():

    def __init__(self):
        self.fc1 = Fc(row = 512, column = 1024)
        self.tanh1 = TanH()
        self.fc2 = Fc(row = 256, column = 512)
        self.tanh2 = TanH()
        self.fc3 = Fc(row = 128 , column = 256)
        self.tanh3 = TanH()
        self.fc4 = Fc(row = 10 , column = 128)
        self.softmax = Softmax()

        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, X):
        #print(X.shape)
        flatten = X.reshape(X.shape[0], -1)
        #print(flatten.shape)
        
        fc1 = self.fc1.forward(flatten)
        act1 = self.tanh1.forward(fc1)
        #print(act1.shape)

        fc2 = self.fc2.forward(act1)
        act2 = self.tanh2.forward(fc2)
        #print(act2.shape)
        
        fc3 = self.fc3.forward(act2)
        act3 = self.tanh3.forward(fc3)
        #print(act3.shape)
        
        fc4 = self.fc4.forward(act3)
        
        y_pred = self.softmax.forward(fc4)
      
        return y_pred

    def backward(self, y_pred, y):
        deltaL = self.softmax.backward(y_pred, y)
        deltaL, dW4, db4 = self.fc4.backward(deltaL)
        deltaL = self.tanh3.backward(deltaL)
        #print(deltaL.shape, dW4.shape, db4.shape)
        
        deltaL, dW3, db3 = self.fc3.backward(deltaL)
        deltaL = self.tanh2.backward(deltaL)

        deltaL, dW2, db2 = self.fc2.backward(deltaL)
        deltaL = self.tanh1.backward(deltaL)

        deltaL, dW1, db1 = self.fc1.backward(deltaL)

        grads = { 
                'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2, 
                'dW3': dW3, 'db3': db3,
                'dW4': dW4, 'db4': db4
        }
        
        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['W' + str(i+1)] = layer.W['val']
            params['b' + str(i+1)] = layer.b['val']

        # params = {      
        #     'W1': self.fc1.W['val'], 'b1': self.fc1.b['val'], 
        #     'W2': self.fc2.W['val'], 'b2': self.fc2.b['val'], 
        #     'W3': self.fc3.W['val'], 'b3': self.fc3.b['val'], 
        #     'W4': self.fc4.W['val'], 'b4': self.fc4.b['val'],
        # }  
        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W['val'] = params['W'+ str(i+1)]
            layer.b['val'] = params['b' + str(i+1)]
      
        # self.fc1.W['val'] = params['W1']
        # self.fc2.W['val'] = params['W2']
        # self.fc3.W['val'] = params['W3'] 
        # self.fc4.W['val'] = params['W4'] 

        # self.fc1.b['val'] = params['b1']
        # self.fc2.b['val'] = params['b2']
        # self.fc3.b['val'] = params['b3'] 
        # self.fc4.b['val'] = params['b4'] 
