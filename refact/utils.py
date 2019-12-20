#------------------
#Utilities function
#------------------

import urllib.request
import gzip
import os
from skimage import transform
import numpy as np

def download_mnist(filename):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for elt in filename:
        print("Downloading " + elt[1] + " in data/ ...")
        urllib.request.urlretrieve(base_url+elt[1], 'data/' + elt[1])
    print("Download complete.")


def extract_mnist(filename):
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
        A function to resize a batch of MNIST images to (32, 32)

        Parameters:
        -imgs: a numpy array of size [batch_size, 28 X 28].
    """
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs