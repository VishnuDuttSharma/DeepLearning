'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Vishnu Dutt Sharma
Roll No.: 12EC35018

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import requests
import os

def downloadData_DNN():
    http_proxy  = "http://10.3.100.207:8080"
    https_proxy = "https://10.3.100.207:8080"
    ftp_proxy   = "ftp://10.3.100.207:8080"

    proxyDict = { 
                  "http"  : http_proxy, 
                  "https" : https_proxy, 
                  "ftp"   : ftp_proxy
                }
    
    base = 'https://raw.githubusercontent.com/VishnuDuttSharma/DL_weights/master/asg3/'
    
    weightList = ['bias_nn_1.npy', 'bias_nn_2.npy', 'bias_nn_out.npy', 'weight_nn_1.npy', 'weight_nn_2.npy', 'weight_nn_out.npy']
    
    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    
    count = 0
    total = len(weightList)
    
    print('Downloading DNN Weights')
    for name in weightList:
        url = base + name
        stream = requests.get(url, proxies=proxyDict)
        np_f = open('./weights/'+name, 'wb')
        np_f.write(stream.content)
        np_f.close()
        print(count+1,'/',total,' Complete')
        count += 1
        
    print('Download Complete')


def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



def train(trainX, trainY):
    trainX_mod = trainX.reshape((trainX.shape[0], trainX.shape[1]*trainX.shape[2]))
#     trainX_mod = trainX_mod - trainX_mod.mean(axis=0)
    y_mod = []
    for i in trainY:
        zer = [0]*10
        zer[i] = 1
        y_mod.append(zer)

    trainY_mod = np.asarray(y_mod)
    
    n_hidden = 400
    n_input = 784
    n_classes = 10

    n_hidden_1 = 400  # 1st layer number of features
    n_hidden_2 = 100# 2nd layer number of features
    n_input = trainX_mod.shape[1] # MNIST data input (img shape: 28*28)
    n_classes = trainY_mod.shape[1] # MNIST total classes (0-9 digits)

    
    
    
    X = tf.placeholder("float", shape=[None, n_input])
    y = tf.placeholder("float", shape=[None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0.001)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0.001)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], 0.001))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0.001)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0.001)),
        'out': tf.Variable(tf.random_normal([n_classes], 0.001))
    }
    
    x = tf.placeholder(tf.float32, shape=[None, n_input])
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    predict = multilayer_perceptron(X, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    updates = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    
    accuracy = 0.0
    batch_size = 100
    iterat = 20

    
    for epoch in range(iterat):
        train_X, test_X, train_y, test_y = train_test_split(trainX_mod, trainY_mod, test_size=0.20, random_state=random.randint(1,99))
        # Train with each example
        for i in range(int(len(train_X)/batch_size)):
            sess.run(updates, feed_dict={X: train_X[i*batch_size: (i + 1)*batch_size], y: train_y[i*batch_size: (i + 1)*batch_size]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 np.argmax(sess.run(predict, feed_dict={X: train_X, y: train_y}), axis=1))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 np.argmax(sess.run(predict, feed_dict={X: test_X, y: test_y}), axis=1))

        accuracy += test_accuracy
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
    
    
    
    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    
    w1= sess.run(weights['h1'])
    np.save('./weights/weight_nn_1', w1)

    w2= sess.run(weights['h2'])
    np.save('./weights/weight_nn_2', w2)
    
    wout= sess.run(weights['out'])
    np.save('./weights/weight_nn_out', wout)

    b1= sess.run(biases['b1'])
    np.save('./weights/bias_nn_1', b1)

    b2= sess.run(biases['b2'])
    np.save('./weights/bias_nn_2', b2)
    
    bout= sess.run(biases['out'])
    np.save('./weights/bias_nn_out', bout)
    
    sess.close()


def test(testX):
    download = True
    if download:
        downloadData_DNN()
        

    testX_mod = testX.reshape((testX.shape[0], testX.shape[1]*testX.shape[2]))
    
    n_hidden = 400
    n_input = 784
    n_classes = 10

    n_hidden_1 = 500  # 1st layer number of features
    n_hidden_2 = 150# 2nd layer number of features
    n_input = testX_mod.shape[1] # MNIST data input (img shape: 28*28)
        
    
    X = tf.placeholder("float", shape=[None, n_input])
    
    weights = dict()
    biases = dict()
    

    weights['h1'] = np.load('./weights/weight_nn_1.npy')
    weights['h2'] = np.load('./weights/weight_nn_2.npy')
    weights['out'] = np.load('./weights/weight_nn_out.npy')

    biases['b1'] = np.load('./weights/bias_nn_1.npy')
    biases['b2'] = np.load('./weights/bias_nn_2.npy')
    biases['out'] = np.load('./weights/bias_nn_out.npy')

    x = tf.placeholder(tf.float32, shape=[None, n_input])
    
    predict = multilayer_perceptron(X, weights, biases)
    
    gpu = False    
    config=tf.ConfigProto()
    if gpu:
        config.gpu_options.allow_growth = True
        
    sess = tf.Session(config=config)

    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    output = np.argmax(sess.run(predict, feed_dict={X: testX_mod}), axis=1)
    
    return output
    
