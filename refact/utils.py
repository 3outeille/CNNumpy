#------------------
#Utilities function
#------------------

import urllib.request
import gzip
import os
from skimage import transform
import numpy as np

def download_mnist(filename):
    """

    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for elt in filename:
        print("Downloading " + elt[1] + " in data/ ...")
        urllib.request.urlretrieve(base_url + elt[1], 'data/' + elt[1])
    print("Download complete.")


def extract_mnist(filename):
    """

    """
    mnist = {}
    for elt in filename[:2]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open('data/' + elt[1]) as f:
            #According to the doc on MNIST website, offset for image starts at 16.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28*28)
    
    for elt in filename[2:]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open('data/' + elt[1]) as f:
            #According to the doc on MNIST website, offset for label starts at 8.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    print('Extraction complete') 

    return mnist

def load(filename):
    """

    """
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

    print('Loading complete')
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
        

def resize_batch(imgs):
    """
        Resizes a batch of MNIST images to (32, 32).

        Parameters:
        -imgs: a numpy array of size [batch_size, 28 X 28].
    """
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


def get_batch(X, batch_size, t):
    """

    """
    return X[t*batch_size : (t + 1)*batch_size]


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
    m, n_H, n_W, n_C = M.shape

    for i in range(m):
        
        for c in range(n_C):
            print('Image {}, channel {}'.format(i + 1, c + 1), end='\n\n')  

            for h in range(n_H):
                print("/", end="")

                for j in range(n_W):

                    print(M[i, h, j, c], end = " ")

                print("/", end='\n\n')
        
        print('-------------------', end='\n\n')

