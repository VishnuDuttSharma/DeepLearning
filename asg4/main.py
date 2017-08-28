
# coding: utf-8


# In[2]:

import tensorflow as tf
import numpy as np
import os
import requests
import ptb_reader


tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test', '../data/ptb.test.txt',"Path to file for testing ")
tf.app.flags.DEFINE_boolean('download', True, 'Choose whether to download data or train the model. Default is --download. Other option is --nodownload')
tf.app.flags.DEFINE_boolean('verbose', False, 'Set verbose to 1 if you want to see the progress and steps. Default is  --noverbose')

# In[3]:

init_limit = 0.05
num_layers = 2
num_steps = 35
hidden_size = 300
max_epoch = 10
keep_prob = 0.5
batch_size = 20
vocab_size = 10000
learning_rate = 0.002


# In[4]:

class TrainModel():
    def __init__(self, reuse = False, is_training = True):     

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_data")
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps], name="targets")


        with tf.variable_scope('model_1', reuse=reuse):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [vocab_size, hidden_size], dtype=tf.float32)
                train_input = tf.nn.embedding_lookup(embedding, self.input_data)

            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
            


            if(is_training):
            	lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

            initializer = tf.random_uniform_initializer(-init_limit, init_limit)



            w_out = tf.get_variable("w_out", [hidden_size, vocab_size], dtype=tf.float32)
            b_out = tf.get_variable("b_out", [vocab_size], dtype=tf.float32)

            self.initial_state = cell.zero_state(batch_size, tf.float32)

            if(is_training):
                train_input_drop = tf.nn.dropout(train_input, keep_prob)
            else:
                train_input_drop = train_input

        
            state =  self.initial_state

            output_arr = []
            state_arr = []
            with tf.variable_scope('RNN', initializer=initializer):
                for step in range(num_steps):
                    if step > 0:
                        tf.get_variable_scope().reuse_variables()

                    (cell_output, state) = cell(train_input_drop[:,step,:], state)

                    output_arr.append(cell_output)
                    state_arr.append(state)

            self.final_state = state_arr[-1]

            
            # output = tf.reshape(output_arr[-1], [-1, hidden_size])
            # targets = tf.one_hot(self.targets[:,-1], vocab_size)

            output = tf.reshape(tf.concat(values=output_arr, axis=1), [-1, hidden_size])
            targets = tf.reshape(self.targets, [-1])
            weights = tf.ones([batch_size * num_steps])
            

            logits = tf.nn.xw_plus_b(output, w_out, b_out)
            

            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights], vocab_size)

            self.cost = cost = tf.reduce_sum(tf.pow(loss,2))/batch_size

            output_state = self.final_state
    
            if( is_training ):
                self.train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[5]:

def downloadData(verbose = 1):
    http_proxy  = "http://10.3.100.207:8080"
    https_proxy = "https://10.3.100.207:8080"
    ftp_proxy   = "ftp://10.3.100.207:8080"

    proxyDict = { 
                  "http"  : http_proxy, 
                  "https" : https_proxy, 
                  "ftp"   : ftp_proxy
                }

    base = 'https://raw.githubusercontent.com/VishnuDuttSharma/DL_weights/master/asg4/'
    
    weightList = ['checkpoint', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta']
    
    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    
    count = 0
    total = len(weightList)
    
    if verbose:
        print('Downloading Weights')
    for name in weightList:
        url = base + name
        stream = requests.get(url, proxies=proxyDict)
        np_f = open('./weights/'+name, 'wb')
        np_f.write(stream.content)
        np_f.close()
        if verbose:
            print(count+1,'/',total,' Complete')
        count += 1
        
    if verbose:
        print('Download Complete')


def get_data(filepath):
    word_to_id = ptb_reader._build_vocab(filepath)
    return ptb_reader._file_to_word_ids(filepath, word_to_id), len(word_to_id)


# In[6]:

def train_model(data, reuse=None, verbose = True):
    epoch_size = ((len(data) // batch_size) - 1)
    costs = 0.0
    iters = 0
    perplexity = 0.0
    
    model = TrainModel(reuse = reuse, is_training=True)
    
    saver = tf.train.Saver()
    
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    state = sess.run(model.initial_state)
    
    
    for i in range(max_epoch):
        for step, (x, y) in enumerate(ptb_reader.ptb_iterator(data, batch_size, num_steps)):
            
            cost, state, _ = sess.run([model.cost, model.final_state, model.train_optimizer], feed_dict={
                model.input_data: x,
                model.targets: y,
                model.initial_state: state
            })
            costs += cost
            iters += 1

            perplexity = cost

            if step % 100 == 0 and verbose:
                print("Epoch: %d, Step: %d, Perplexity: %.3f  " % (i, step, perplexity))
            
    saver.save(sess, './weights/model.ckpt')
    sess.close()
    tf.reset_default_graph()
    return (costs / iters), True
    


# In[7]:

def test_model(data, verbose = True, reuse=None):
    epoch_size = ((len(data) // batch_size) - 1)
    costs = 0.0
    iters = 0

    # with tf.variable_scope("model_2", reuse=reuse):
    model = TrainModel( is_training=False)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # model = None

    new_saver = tf.train.import_meta_graph('./weights/model.ckpt.meta')
    new_saver.restore(sess, './weights/model.ckpt')
    state = sess.run(model.initial_state)
    
    perplexity = 0.0

    for step, (x, y) in enumerate(ptb_reader.ptb_iterator(data, batch_size, num_steps)):
#         print()
        cost, state = sess.run([model.cost, model.final_state], feed_dict={
            model.input_data: x,
            model.targets: y
#             model.initial_state: state
        })
        perplexity += cost
        iters += 1


        if step % 100 == 0 and verbose:
            progress = step
            print("%d Perplexity: %.3f " % (progress, cost))

    sess.close()

    return (perplexity / iters)
    


# In[13]:

def main(_):
    global vocab_size
    model_exist = False

    if FLAGS.download:
        downloadData(verbose = FLAGS.verbose)        
    else:
        train_data, feat_len = get_data('../data/ptb.train.txt')
        vocab_size = feat_len
        model = None
        model, model_exist = train_model(train_data, verbose = FLAGS.verbose)
        

    test_data, feat_len = get_data(FLAGS.test)
    vocab_size = feat_len
    perp = test_model(test_data, verbose = FLAGS.verbose)
    
    print(perp)


# In[14]:


##	Run python main.py --help for options
##
if __name__ == "__main__":
    tf.app.run()



