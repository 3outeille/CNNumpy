#------------------
#Utilities function
#------------------

import urllib.request
import gzip
import os
from skimage import transform
from PIL import Image
import numpy as np
import pickle

def download_mnist(filename):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for elt in filename:
        print("Downloading " + elt[1] + " in data/ ...")
        urllib.request.urlretrieve(base_url + elt[1], 'data/' + elt[1])
    print("Download complete.")


def extract_mnist(filename):
    mnist = {}
    for elt in filename[:2]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open('data/' + elt[1]) as f:
            #According to the doc on MNIST website, offset for image starts at 16.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28)
    
    for elt in filename[2:]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open('data/' + elt[1]) as f:
            #According to the doc on MNIST website, offset for label starts at 8.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    print('Files extraction: OK') 

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

    print('Loading dataset: OK')
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
        

def resize_dataset(dataset):
    """
        Resizes dataset of MNIST images to (32, 32).

        Parameters:
        -dataset: a numpy array of size [?, 1, 28, 28].
    """
    return transform.resize(dataset, (dataset.shape[0], 1, 32, 32))

def dataloader(X, y, BATCH_SIZE):
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t+BATCH_SIZE, ...], y[t:t+BATCH_SIZE, ...]
        
def one_hot_encoding(y):
    N = y.shape[0]
    Z = np.zeros((N, 10))
    Z[np.arange(N), y] = 1
    return Z

def train_val_split(X, y):
    """
        Splits X and y into training and validation set.
    """
    X_train, X_val = X[:50000, :], X[50000:, :]
    y_train, y_val = y[:50000, :], y[50000:, :]

    return X_train, y_train, X_val, y_val


def load_params_from_file(model, filename):
    """
        Loads model parameters from a file.

        Parameters:
        -model: a CNN architecture.
        -filename: name of file with extension 'pkl'.
    """
    pickle_in = open(filename, 'rb')
    params = pickle.load(pickle_in)

    model.conv1.W['val'] = params['W1']
    model.conv2.W['val'] = params['W2']
    model.fc1.W['val'] = params['W3']
    model.fc2.W['val'] = params['W4']
    model.fc3.W['val'] = params['W5'] 

    model.conv1.b['val'] = params['b1']
    model.conv2.b['val'] = params['b2']
    model.fc1.b['val'] = params['b3']
    model.fc2.b['val'] = params['b4']
    model.fc3.b['val'] = params['b5'] 

def save_params_to_file(model, filename):
    """
        Saves model parameters to a file.

        Parameters:
        -model: a CNN architecture.
        -filename: name of file with extension 'pkl'.
    """
    weights = model.get_params()
    with open(filename,"wb") as f:
	    pickle.dump(weights, f)

def prettyPrint3D(M):
    """
        Displays a 3D matrix in a pretty way.

        Parameters:
        -M: Matrix of shape (m, n_H, n_W, n_C) with m, the number 3D matrices.
    """
    m, n_C, n_H, n_W = M.shape

    for i in range(m):
        
        for c in range(n_C):
            print('Image {}, channel {}'.format(i + 1, c + 1), end='\n\n')  

            for h in range(n_H):
                print("/", end="")

                for j in range(n_W):

                    print(M[i, c, h, j], end = ",")

                print("/", end='\n\n')
        
        print('-------------------', end='\n\n')


