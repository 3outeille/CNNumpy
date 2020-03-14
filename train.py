from utils import *
from model import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def train(filename):
    NB_EPOCH = 1
    ITER = 938
    BATCH_SIZE = 64

    X, y, X_test, y_test = load(filename)
    X, X_test = X/float(255), X_test/float(255)
    X -= np.mean(X)
    X_test -= np.mean(X_test)

    X_train, y_train, X_val, y_val = train_val_split(X, y)

    model = LeNet5()
    cost = CrossEntropyLoss()
    optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    
    
    costs = []

    for epoch in range(NB_EPOCH):

        for t in range(ITER):
            X_batch = resize_batch(get_batch(X_train, BATCH_SIZE, t))
            y_batch = get_batch(y_train, BATCH_SIZE, t)
            y_batch_encoded = one_hot_encoding(y_batch)
    
            y_pred = model.forward(X_batch)
            loss, deltaL = cost.get(y_pred, y_batch_encoded)
            
            grads = model.backward(deltaL)
            params = optimizer.update_params(grads)
            model.set_params(params)

            costs.append(loss)

            accuracy = accuracy_score(y_batch, np.argmax(y_pred, axis=0))

            print('[Epoch {} | ITER {} / {}] Loss: {} | Accuracy: {}'.format(epoch+1, t+1, ITER, loss, accuracy))

    plt.plot(costs)
    plt.show()
    save_params_to_file(model, "final_weights.pkl")

train(filename)
