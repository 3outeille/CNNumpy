from utils import *
from model import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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

    print("Preprocessing step:")
    X, X_test = resize_dataset(X), resize_dataset(X_test)
    y, y_test = one_hot_encoding(y), one_hot_encoding(y_test)
    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print("Preprocessing step done.")

    train_loader = dataloader(X_train, y_train, BATCH_SIZE)

    model = LeNet5()
    cost = CrossEntropyLoss()
    optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    
    
    costs = []
    
    for epoch in range(NB_EPOCH):

        for t, (X_batch, y_batch) in enumerate(train_loader):
    
            y_pred = model.forward(X_batch)
            loss, deltaL = cost.get(y_pred, y_batch)
            
            grads = model.backward(deltaL)
            params = optimizer.update_params(grads)
            model.set_params(params)

            costs.append(loss)

            print(y_batch)
            print(y_batch.shape)
            print("------------")
            print(y_pred)
            print(y_pred.shape)
            print("------------")
            y_pred_encoded = np.argmax(y_pred, axis=0)
            print(y_pred_encoded)
            print(y_pred_encoded.shape)
            return 
        
            #accuracy = accuracy_score(y_batch, np.argmax(y_pred, axis=0))

            print('[Epoch {} | ITER {} / {}] Loss: {} | Accuracy: {}'.format(epoch+1, t+1, ITER, loss, accuracy))

    plt.plot(costs)
    plt.show()
    save_params_to_file(model, "final_weights.pkl")

train(filename)
