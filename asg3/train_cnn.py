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

def downloadData_CNN():
    http_proxy  = "http://10.3.100.207:8080"
    https_proxy = "https://10.3.100.207:8080"
    ftp_proxy   = "ftp://10.3.100.207:8080"

    proxyDict = { 
                  "http"  : http_proxy, 
                  "https" : https_proxy, 
                  "ftp"   : ftp_proxy
                }

    base = 'https://raw.githubusercontent.com/VishnuDuttSharma/DL_weights/master/asg3/'
    
    weightList = ['b_conv1.npy', 'b_conv2.npy', 'b_fc1.npy', 'b_fc2.npy',  'W_conv1.npy', 'W_conv2.npy', 'W_fc1.npy', 'W_fc2.npy']
    
    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    
    count = 0
    total = len(weightList)
    
    print('Downloading CNN Weights')
    for name in weightList:
        url = base + name
        stream = requests.get(url, proxies=proxyDict)
        np_f = open('./weights/'+name, 'wb')
        np_f.write(stream.content)
        np_f.close()
        print(count+1,'/',total,' Complete')
        count += 1
        
    print('Download Complete')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def train(trainX, trainY):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x = tf.placeholder(tf.float32, [None,28,28,1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    y_mod = []
    for i in trainY:
        zer = [0]*10
        zer[i] = 1
        y_mod.append(zer)

    train_Y_mod = np.asarray(y_mod)
    
    
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    gpu = False    
    config=tf.ConfigProto()
    if gpu:
        config.gpu_options.allow_growth = True
        
    sess = tf.Session(config=config)

    # sess = tf.InteractiveSession()

    sess.run(init)
    
    
    
    epoch = 0
    batch_size = 100
    
    
    while epoch < 3:
        step = 1
        train_X, test_X, train_y, test_y = train_test_split(trainX, train_Y_mod, test_size=0.20, random_state=random.randint(1,99))
        # Keep training until reach max iterations
        for k in range(int(len(train_X)/batch_size)):
#             print("Range: ", k*batch_size, " to ", (k+1)*batch_size)
            
            batch_x, batch_y = train_X[k*batch_size : (k+1)*batch_size], train_y[k*batch_size : (k+1)*batch_size]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
            
            if step % 100 == 0:
                print("Epoch: %d, step %d, training accuracy %g"%(epoch, k, train_accuracy*100))
            
            
            step += 1
        
        acc = 0.0
        for k in range(int(len(test_X)/batch_size)):
            batch_x, batch_y = test_X[k*batch_size : (k+1)*batch_size], test_y[k*batch_size : (k+1)*batch_size]
            acc += sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        
        print(" test accuracy, ", str(100*acc/int(len(test_X)/batch_size)))
        
        epoch += 1
            
    
    np.save('./weights/W_conv1', sess.run(W_conv1))
    np.save('./weights/W_conv2', sess.run(W_conv2))
    np.save('./weights/W_fc1', sess.run(W_fc1))
    np.save('./weights/W_fc2', sess.run(W_fc2))
    
    np.save('./weights/b_conv1', sess.run(b_conv1))
    np.save('./weights/b_conv2', sess.run(b_conv2))
    np.save('./weights/b_fc1', sess.run(b_fc1))
    np.save('./weights/b_fc2', sess.run(b_fc2))

    sess.close()

def test(testX):
    download = True
    if download:
        downloadData_CNN()
    
    W_conv1 = np.load('./weights/W_conv1.npy')
    b_conv1 = np.load('./weights/b_conv1.npy')

    W_conv2 = np.load('./weights/W_conv2.npy')
    b_conv2 = np.load('./weights/b_conv2.npy')

    W_fc1 = np.load('./weights/W_fc1.npy')
    b_fc1 = np.load('./weights/b_fc1.npy')

    W_fc2 = np.load('./weights/W_fc2.npy')
    b_fc2 = np.load('./weights/b_fc2.npy')
        
    x = tf.placeholder(tf.float32, [None,28,28,1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    gpu = False    
    config=tf.ConfigProto()
    if gpu:
        config.gpu_options.allow_growth = True
        
    sess = tf.Session(config=config)

    sess.run(init)
    
    output = []
    batch_size = 100
    
    for k in range((int(len(testX)/batch_size))):
        out = sess.run(y_conv, feed_dict={x: testX[k*batch_size : (k+1)*batch_size], keep_prob: 1.0})
        output += list(sess.run(tf.arg_max(out,1)))
        
        
    sess.close()
    
    return output
