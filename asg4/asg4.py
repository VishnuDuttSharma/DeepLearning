
# coding: utf-8

# In[ ]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[1]:

import tensorflow as tf
import ptb_reader
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import load_model

import os


# In[ ]:




# In[2]:

batch_size = 1000
dataset_size = 10000
num_steps = 10
hidden_size = 128
feat_len = 10000


# In[3]:

def perplexity(y_true, y_pred):
    cross_ent_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_pred, logits=y_true)
    return tf.pow(cross_ent_1, 2)


# In[4]:

def one_hot(x, y):
    xmat = np.zeros((x.shape[0], num_steps, feat_len))
    for j in range(x.shape[1]):
        for i in zip(range(num_steps), x[j]):
            xmat[j, i[0], i[1]] = 1
    
    ymat = np.zeros((x.shape[0], feat_len))
    for j in range(x.shape[1]):
        ymat[j, y[j]] = 1
    return xmat, ymat


# In[16]:

def get_model():
    
    model = Sequential()
    # , dropout=0.2, recurrent_dropout=0.2
    model.add(LSTM(hidden_size,  input_shape=( num_steps, feat_len)))
#     model.add(LSTM(hidden_size))
    model.add(Dense(feat_len, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['categorical_accuracy'])
    
    return model


# In[6]:

# model.add(Embedding(max_features, num_steps))
# model.add(LSTM(50, input_shape=(1,1), dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(vocab, activation='softmax'))
# model.pop()
# model.add(Dense(10, input_shape=(10,), activation='softmax'))

# model.pop()
# model.pop()





# plot_model(model, to_file='model.png')


# In[7]:

def get_data(filepath):
    word_to_id = ptb_reader._build_vocab(filepath)
    return ptb_reader._file_to_word_ids(filepath, word_to_id), len(word_to_id)


# In[19]:


def train(train_data, verbose = 0, model = None):
    if model is None:
        model = get_model()
    
    if(verbose > 0):
        print('Train...')

    for step, (x, y) in enumerate(ptb_reader.ptb_iterator(train_data, dataset_size, num_steps)):    
#         x1 = np.zeros((dataset_size, num_steps, feat_len))
#         for i in range(x1.shape[0]):
#             for j in range(x1.shape[1]):
#                 x1[i,j,x[i,j]] = 1

#         y1 = y[:,-1]

#         y1 = np.zeros((dataset_size, feat_len))
#         for i in range(y1.shape[0]):
#             y1[i,y[i,-1]] = 1

        x1, y1 = one_hot(x, y[:, -1])
        
        model.fit(x1, y1, epochs=10, verbose = verbose, batch_size=10)

        if(step % 100 == 0 and verbose> 0):
            print(step+1, end=' ')
        break
        
    if( not os.path.isdir('weights') ):
        os.mkdir('weights')
        
    model.save('weights/my_model.h5')
    
#     model.save_weights('my_model_weights.h5')
    
    


# In[24]:

def test(test_data, verbose = 0):

    model =load_model('weights/my_model.h5')

#     model = get_model()
#     model.load_weights('my_model_weights.h5')
    
    acc = 0.0
    siz = 0
    perplexity = []
    for step, (x, y) in enumerate(ptb_reader.ptb_iterator(test_data, dataset_size, num_steps)):
        
        x1, y1 = one_hot(x,y[:,-1])

        output = model.predict(x1, verbose=verbose)
        score, accuracy  = model.evaluate(x1, y1, verbose = 1, batch_size=10)
 
        perplexity.append(np.power(accuracy,2))
                        
        siz += 1 
        
        
        print('')
        print('Step: ',step+1, end='')
        print(', Test accuracy:', accuracy )
        
        acc += accuracy

    
        
    print('Average Accuracy: ', acc/siz)
    
    return np.mean(perp_np)
    
    


# In[17]:




# In[11]:

if __name__ == "__main__":
    global feat_len
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('test', '../data/ptb.test.txt',
                       """Path to file for testing """)

#     train_data, valid_data, test_data, vocab = ptb_reader.ptb_raw_data("../data")
    train_data, feat_len = get_data('../data/ptb.train.txt')
    model = None
    model = train(train_data, verbose = 1, model = model)
    
    test_data, _ = get_data(FLAGS.test)
    perp =test(test_data)
    print(perp)
    


# In[23]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



