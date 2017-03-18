'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Vishnu Dutt Sharma
Roll No.: 12EC35018


======================================

Problem Statement:
This assignment has two problems.

P1. Implement a simple 1 or 2 hidden layer MLP USING any deep learning library
for predicting MNIST images.
P2. Implement a simple CNN USING any deep learning library
for predicting MNIST images.

Resources:
1. https://www.tensorflow.org/get_started/mnist/beginners
2. https://www.tensorflow.org/get_started/mnist/pros

======================================

Instructions:
1. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/
    (four files).
2. Extract all the files into a folder named `data' just outside
    the folder containing the main.py file. This code reads the
    data files from the folder '../data'.
3. Complete the functions in the train_dense.py and train_cnn.py files. You might also
    create other functions for your convenience, but do not change anything
    in the main.py file or the function signatures of the train and test
    functions in the train files.
4. The train function must train the neural network given the training
    examples and save the in a folder named `weights' in the same
    folder as main.py
5. The test function must read the saved weights and given the test
    examples it must return the predicted labels.
6. Submit your project folder with the weights. Note: Don't include the
    data folder, which is anyway outside your project folder.

Submission Instructions:
1. Fill your name and roll no in the space provided above.
2. Name your folder in format <Roll No>_<First Name>.
    For example 12CS10001_Rohan
3. Submit a zipped format of the file (.zip only).
'''

import numpy as np
import os
import train_dense
import train_cnn


def load_mnist():
    data_dir = '../data'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY


def print_digit(digit_pixels, label='?'):
    for i in range(28):
        for j in range(28):
            if digit_pixels[i, j] > 128:
                print('#',end='')
            else:
                print('.',end='')
        print('')

    print('Label: ', label)


def main():
    trainX, trainY, testX, testY = load_mnist()
    print("Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape)

    print("\nDigit sample")
    print_digit(trainX[1], trainY[1])

    train_dense.train(trainX, trainY)
    labels = train_dense.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print("\nDNN Test accuracy: %lf%%" % accuracy)

    train_cnn.train(trainX, trainY)
    labels = train_cnn.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print("\nCNN Test accuracy: %lf%%" % accuracy)


if __name__ == '__main__':
    main()
