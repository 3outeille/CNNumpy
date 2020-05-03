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

def predict():
    X_train, y_train, X_test, y_test = load(filename)
    X_train, X_test = X_train/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    model = LeNet5()
    load_params_from_file(model,'final_weights.pkl')
    
    image = resize_batch(X_train[0])
    y_pred = model.forward(image)
    tmp = np.argmax(y_pred, axis=0)
    print(tmp)

    first_image = image
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((32, 32))
    plt.imshow(pixels, cmap='gray')
    plt.show()

predict()