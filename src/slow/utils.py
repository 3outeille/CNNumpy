from src.slow.data import *
import concurrent.futures as cf
import urllib.request
import gzip
import os
from skimage import transform
from PIL import Image
import numpy as np
import math
import pickle

def download_mnist(filename):
    """
        Downloads dataset from filename.

        Parameters:
        - filename: [
                        ["training_images","train-images-idx3-ubyte.gz"],
                        ["test_images","t10k-images-idx3-ubyte.gz"],
                        ["training_labels","train-labels-idx1-ubyte.gz"],
                        ["test_labels","t10k-labels-idx1-ubyte.gz"]
                    ]
    """
    # Make data/ accessible from every folders.
    terminal_path = ['src/slow/data/', 'slow/data/', 'data/', '../data']
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("extract_mnist(): Impossible to find data/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    base_url = "http://yann.lecun.com/exdb/mnist/"
    for elt in filename:
        print("Downloading " + elt[1] + " in data/ ...")
        urllib.request.urlretrieve(base_url + elt[1], dirPath + elt[1])
    print("Download complete.")

def extract_mnist(filename):
    """
        Extracts dataset from filename.

        Parameters:
        - filename: [
                        ["training_images","train-images-idx3-ubyte.gz"],
                        ["test_images","t10k-images-idx3-ubyte.gz"],
                        ["training_labels","train-labels-idx1-ubyte.gz"],
                        ["test_labels","t10k-labels-idx1-ubyte.gz"]
                    ]
    """
    # Make data/ accessible from every folders.
    terminal_path = ['src/slow/data/', 'slow/data/', 'data/', '../data']
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("extract_mnist(): Impossible to find data/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    mnist = {}
    for elt in filename[:2]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open(dirPath + elt[1]) as f:
            #According to the doc on MNIST website, offset for image starts at 16.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28)
    
    for elt in filename[2:]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open(dirPath + elt[1]) as f:
            #According to the doc on MNIST website, offset for label starts at 8.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    print('Files extraction: OK') 

    return mnist

def load(filename):
    """
        Loads dataset to variables.

        Parameters:
        - filename: [
                        ["training_images","train-images-idx3-ubyte.gz"],
                        ["test_images","t10k-images-idx3-ubyte.gz"],
                        ["training_labels","train-labels-idx1-ubyte.gz"],
                        ["test_labels","t10k-labels-idx1-ubyte.gz"]
                  ]
    """
    # Make data/ accessible from every folders.
    terminal_path = ['src/slow/data/', 'slow/data/', 'data/', '../data']
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("extract_mnist(): Impossible to find data/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    L = [elt[1] for elt in filename]   
    count = 0 

    #Check if the 4 .gz files exist.
    for elt in L:
        if os.path.isfile(dirPath + elt):
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
    args = [dataset[i:i+1000] for i in range(0, len(dataset), 1000)]
    
    def f(chunk):
        return transform.resize(chunk, (chunk.shape[0], 1, 32, 32))

    with cf.ThreadPoolExecutor() as executor:
        res = executor.map(f, args)
    
    res = np.array([*res])
    res = res.reshape(-1, 1, 32, 32)
    return res


def dataloader(X, y, BATCH_SIZE):
    """
        Returns a data generator.

        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t+BATCH_SIZE, ...], y[t:t+BATCH_SIZE, ...]
        
def one_hot_encoding(y):
    """
        Performs one-hot-encoding on y.
        
        Parameters:
        - y: ground truth labels.
    """
    N = y.shape[0]
    Z = np.zeros((N, 10))
    Z[np.arange(N), y] = 1
    return Z

def train_val_split(X, y, val=50000):
    """
        Splits X and y into training and validation set.

        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    X_train, X_val = X[:val, :], X[val:, :]
    y_train, y_val = y[:val, :], y[val:, :]

    return X_train, y_train, X_val, y_val

def save_params_to_file(model):
    """
        Saves model parameters to a file.

        Parameters:
        -model: a CNN architecture.
    """
    # Make save_weights/ accessible from every folders.
    terminal_path = ["src/slow/save_weights/", "slow/save_weights/", "save_weights/", "../save_weights/"]
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("save_params_to_file(): Impossible to find save_weights/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    weights = model.get_params()
    with open(dirPath + "final_weights.pkl","wb") as f:
	    pickle.dump(weights, f)

def load_params_from_file(model):
    """
        Loads model parameters from a file.

        Parameters:
        -model: a CNN architecture.
    """
    # Make final_weights.pkl file accessible from every folders.
    terminal_path = ["src/slow/save_weights/final_weights.pkl", "slow/save_weights/final_weights.pkl",
    "save_weights/final_weights.pkl", "../save_weights/final_weights.pkl"]

    filePath = None
    for path in terminal_path:
        if os.path.isfile(path):
            filePath = path
    if filePath == None:
        raise FileNotFoundError('load_params_from_file(): Cannot find final_weights.pkl from your current folder. You need to manually add it to terminal_path list and the run the function again.')

    pickle_in = open(filePath, 'rb')
    params = pickle.load(pickle_in)
    model.set_params(params)
    return model
            
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
