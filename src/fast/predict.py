from utils import *
from model import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import trange
#from sklearn.metrics import accuracy_score

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def test():
    print("\n--------------------EXTRACTION-------------------\n")
    X, y, X_test, y_test = load(filename)
    X, X_test = X/float(255), X_test/float(255)
    X -= np.mean(X)
    X_test -= np.mean(X_test)

    print("\n------------------PREPROCESSING------------------\n")
    X_test = resize_dataset(X_test)
    print("Resize dataset: OK")
    y_test = one_hot_encoding(y_test)
    print("One-Hot-Encoding: OK")
   
    print("\n--------------LOAD PRETRAINED MODEL--------------\n")
    cost = CrossEntropyLoss()
    model = LeNet5()
    model = load_params_from_file(model, "save_weights/final_weights.pkl")
    print("Load pretrained model: OK\n")

    print("--------------------EVALUATION-------------------\n")
    
    BATCH_SIZE = 128

    nb_test_examples = len(X_test)
    test_loss = 0
    test_acc = 0 

    pbar = trange(nb_test_examples // BATCH_SIZE)
    test_loader = dataloader(X_test, y_test, BATCH_SIZE)

    for i, (X_batch, y_batch) in zip(pbar, test_loader):
      
        y_pred = model.forward(X_batch)
        loss, deltaL = cost.get(y_pred, y_batch)

        test_loss += loss * BATCH_SIZE
        test_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))

        pbar.set_description("Evaluation")
    
    test_loss /= nb_test_examples
    test_acc /= nb_test_examples

    info_test = "test-loss: {:0.6f} | test-acc: {:0.3f}"
    print(info_test.format(test_loss, test_acc))

test()