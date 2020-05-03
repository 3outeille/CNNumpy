from utils import *
from nn import *
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
    NB_EPOCH = 5
    ITER = 938
    BATCH_SIZE = 64

    X, y, X_test, y_test = load(filename)
    X, X_test = X/float(255), X_test/float(255)
    X -= np.mean(X)
    X_test -= np.mean(X_test)
    
    # print(y.shape, y_test.shape)
    # print(y[0], y_test[0])
    # y, y_test = one_hot_encoding(y), one_hot_encoding(y_test)
    # print(y.shape, y_test.shape)
    # print(y[0, ...], y_test[0, ...])
    # print(y[0, ...].shape, y_test[0, ...].shape)
    # return 

    print("Preprocessing step:")
    X, X_test = resize_dataset(X), resize_dataset(X_test)
    y, y_test = one_hot_encoding(y), one_hot_encoding(y_test)
    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print("Preprocessing step done.")

    model = NN()
    cost = CrossEntropyLoss()
    optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    
    #optimizer = SGD(lr=0.001, params=model.get_params())
    costs = []
    

    for epoch in range(NB_EPOCH):
        
        train_loader = dataloader(X_train, y_train, BATCH_SIZE)

        for t, (X_batch, y_batch) in enumerate(train_loader):
        #X_batch, y_batch = X_train[:1, ...], y_train[:1, ...]
        
            y_pred = model.forward(X_batch)
            loss, deltaL = cost.get(y_pred, y_batch)
            
            grads = model.backward(deltaL)
            params = optimizer.update_params(grads)
            model.set_params(params)

            costs.append(loss)

            # print(y_batch)
            # print(y_batch.shape)
            # print(y_pred)
            # print(y_pred.shape)
            # print(np.argmax(y_batch, axis=1))
            # print(np.argmax(y_pred, axis=1))
            # return
            accuracy = accuracy_score(np.argmax(y_batch, axis=1), np.argmax(y_pred, axis=1))

            print('[Epoch {} | ITER {} / {}] Loss: {} | Accuracy: {}'.format(epoch+1, t+1, ITER, loss, accuracy))
        
    plt.plot(costs)
    plt.show()
    #save_params_to_file(model, "final_weights.pkl")

train(filename)
