# -*- coding: utf-8 -*-




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import random
import sys
import os
#sys.path.append('cells')

#import rnn_cell_modern as mcell

import data as data_util
from config import FLAGS


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def _linear(input_, output_size , scope = None):
   shape = input_.get_shape().as_list()
   if len(shape) != 2:
      raise ValueError('Linear is expecting 2D arguments:%s' %str(shape))
   if not shape[1]:
      raise ValueError('Linear expects shape[1] of arguments:%s' %str(shape))
   input_size = shape[1]
   with tf.variable_scope(scope or "SimpleLinear"):
      matrix = tf.get_variable('Matrix', [output_size, input_size], dtype=input_.dtype)
      bias_term = tf.get_variable('bias',[output_size], dtype=input_.dtype)
   return tf.matmul(input_,tf.transpose(matrix))+bias_term


def highway(input_ , size , layer_size = 1 , bias = -2 , f = tf.nn.relu ):
   output = input_
   for idx in xrange(layer_size):
      output = f(_linear(output,size,scope='output_lin_%d'%idx))

      transform_gate = tf.sigmoid(_linear(input_,size,scope='transform_lin_%d'%(idx+bias)))

      carry_gate = 1. - transform_gate

      output = transform_gate * output + carry_gate * input_
   return output
   

class rnn_model(object):
   def __init__( self , vocab_size , batch_size, seq_max_len, lstm_size, cat_num , embedding_size , filter_size , num_filters , cell_type = ''):
      self.vocab_size = vocab_size
      self.batch_size = batch_size
      self.seq_max_len = seq_max_len
      self.lstm_size = lstm_size
      self.cat_num = cat_num
      self.embedding_size = embedding_size
      self.filter_size = filter_size
      self.num_filters = num_filters

      


      self.xx = tf.placeholder(tf.int32 , [ None , self.seq_max_len])
      self.target = tf.placeholder(tf.float32 , [ None , self.cat_num])
      self.seq_len = tf.placeholder(tf.int32 , [ None ])

      self.dropout = tf.placeholder(tf.float32)

      #with tf.device('/cpu:0'):
      #   embedding = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0))
      #   self.x = tf.nn.embedding_lookup(embedding , self.xx)


      #self.x = tf.transpose(self.xx,[1,0,2])
      #self.x = tf.reshape(self.x,[-1,1])
      #self.x = tf.split(0,self.seq_max_len,self.x) 

      with tf.device('/cpu:0'):
         embedding = tf.Variable(tf.random_uniform([self.vocab_size , self.embedding_size], -1.0, 1.0))
         self.x = tf.nn.embedding_lookup( embedding , self.xx )

      with tf.device('/cpu:0'):
         embedding1 = tf.Variable(tf.random_uniform([self.vocab_size , self.embedding_size], -1.0 , 1.0))
         self.x1 = tf.nn.embedding_lookup( embedding1 , self.xx)

      self.learning_rate_v = tf.Variable(float(FLAGS.learning_rate), trainable = False )
      self.learning_rate_decay_op = self.learning_rate_v.assign(self.learning_rate_v * FLAGS.learning_rate_decay_factor)


      self.cell = tf.contrib.rnn.GRUCell(self.lstm_size)
      if cell_type is 'lstm':
         self.cell = tf.contrib.rnn.LSTMCell(self.lstm_size , state_is_tuple = True)
      #elif cell_type is 'hwrnn':
      #   self.cell = mcell.HighwayRNNCell(self.lstm_size)
      #elif cell_type is 'bgc':
      #   self.cell = mcell.BasicGatedCell(self.lstm_size)
      #elif cell_type is 'mgu':
      #   self.cell = mcell.MGUCell(self.lstm_size)
      #elif cell_type is 'lstm_memarr':
      #   self.cell = mcell.LSTMCell_MemoryArray(self.lstm_size)
      #elif cell_type is 'JZS1':
      #   self.cell = mcell.JZS1Cell(self.lstm_size)
      #elif cell_type is 'JZS2':
      #   self.cell = mcell.JZS2Cell(self.lstm_size)
      #elif cell_type is 'JZS3':
      #   self.cell = mcell.JZS3Cell(self.lstm_size)

      self.cell = tf.contrib.rnn.DropoutWrapper(self.cell , output_keep_prob = self.dropout)
      self.lstm_output, states = tf.nn.dynamic_rnn(self.cell, self.x1, dtype = tf.float32)

      self.lstm_output = tf.stack(self.lstm_output)
      self.lstm_output = tf.transpose(self.lstm_output,[1,0,2])
      self.b_size = tf.shape(self.lstm_output)[1]
      self.lstm_output = tf.reshape(tf.slice(self.lstm_output, [self.seq_max_len-1,0,0] , [1, self.b_size , self.lstm_size]),[self.b_size,self.lstm_size])
      #self.lstm_output = tf.nn.dropout(self.lstm_output , self.dropout)

      #self.b_size = tf.shape(self.output)[0]

      #self.index = tf.range(0,self.b_size)*self.seq_max_len + (self.seq_len-1)
      #self.index = tf.range(0,self.b_size)

      #self.output = tf.gather(tf.reshape(output,[-1,self.lstm_size]),index)

      #output = highway(output, self.lstm_size)

      self.xxx = tf.expand_dims( self.x , -1)

      self.pooled_outputs = []
      for filter_size, num_filter in zip(self.filter_size, self.num_filters):
         filter_shape = [filter_size , self.embedding_size , 1 , num_filter]
         W = tf.Variable(tf.truncated_normal(filter_shape , stddev = 0.1))
         b = tf.Variable(tf.constant(0.1, shape = [num_filter]))
         conv = tf.nn.conv2d(self.xxx, W , strides = [1,1,1,1], padding = 'VALID')
         h = tf.nn.relu(tf.nn.bias_add(conv,b))
         pooled = tf.nn.max_pool(h, ksize = [1, self.seq_max_len - filter_size + 1 ,1 , 1] , strides = [ 1,1,1,1], padding = 'VALID')
         self.pooled_outputs.append(pooled)
      self.num_filters_total = sum(num_filters)
      self.h_pool = tf.concat(axis=3, values=self.pooled_outputs)
      self.h_pool_flat = tf.reshape(self.h_pool,[-1,self.num_filters_total])
      self.h_highway = highway(self.h_pool_flat , self.h_pool_flat.get_shape()[1],1,0)
      self.h_highway = tf.nn.dropout(self.h_highway , self.dropout)
      

      self.output = tf.concat(axis=1, values=[self.lstm_output , self.h_highway])

      weights = {'out':tf.Variable(tf.random_normal([self.lstm_size + self.num_filters_total , self.cat_num]))}
      biases = {'out':tf.Variable(tf.random_normal([self.cat_num]))}

      self.pred = tf.matmul(self.output, weights['out']) + biases['out']
      self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.target))
      self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_v).minimize(self.cost)
      #self.ppp = tf.argmax(self.pred,1)
      self.correct_pred = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.target,1))

   def step(self, sess, in_data, target, seq_len, dropout):
      return sess.run( [ self.learning_rate_decay_op , self.optimizer], {self.xx:in_data, self.target:target, self.seq_len:seq_len , self.dropout:dropout})

   #def test_acc(self, sess, in_data, target, seq_len, dropout):
   #   return sess.run(self.pred , {self.xx: in_data, self.target:target , self.dropout:dropout})

   #def get_embedding(self, sess , data):
   #   return sess.run(self.embeddings , {self.xx: data})

   def test_acc(self, sess, in_data, target, seq_len, dropout):
      return sess.run([self.correct_pred], {self.xx: in_data, self.target:target, self.seq_len: seq_len, self.dropout:dropout})


   def test_data(self, data, target):
      re_x = []
      re_y = target
      re_seq_len = []
      for i in data:
         t_x = i

         # pad input
         if len(t_x) >= self.seq_max_len:
            t_x = t_x[0:self.seq_max_len-1]
         t_seq_len = len(t_x)
         t_x.append(data_util.EOS_ID)
         t_x = t_x + [data_util.PAD_ID]*(self.seq_max_len - t_seq_len - 1 )


         # one hot
         #t_x = [0]*self.vocab_size
         #for j in i:
         #   t_x[j] = t_x[j] + 1
         #t_x = [[x] for x in t_x]
         re_x.append(t_x)
         #re_x.append([[xx] for xx in t_x])
         re_seq_len.append(t_seq_len)
      return re_x , re_y, re_seq_len

   def next_random_batch(self, train, target):
      batch_x = []
      batch_y = []
      batch_seq_len = []
      for _ in range(self.batch_size):
         r_num = random.randint(0, len(train)-1 )
         t_x = train[r_num]
         t_y = target[r_num]
         # pad input
         if len(t_x) >= self.seq_max_len:
            t_x = t_x[0:(self.seq_max_len-1)]
         t_seq_len = len(t_x)
         t_x.append(data_util.EOS_ID)
         t_x = t_x + [data_util.PAD_ID]*(self.seq_max_len - t_seq_len - 1)
         #t_x = [[x] for x in t_x]
         #print(np.array(t_x).shape)

         #t_x = [0]*self.vocab_size
         #for i in train[r_num]:
         #   t_x[i] = t_x[i] + 1

         #t_x = [[x] for x in t_x]
            
         batch_x.append(t_x)
         batch_y.append(t_y)
         batch_seq_len.append(t_seq_len)
      #print(np.array(batch_x).shape)
      #print(np.array(batch_x).ndim)
      #for i in batch_x:
      #   print(i)
      return batch_x, batch_y, batch_seq_len


def main(_):
   data_util.create_vocab(FLAGS.train, FLAGS.vocab, FLAGS.cat, vocab_size = FLAGS.vocab_max_size)
   vocab, vocab_re = data_util.init_vocab(FLAGS.vocab)
   vocab_size = len(vocab)
   cat, cat_re = data_util.init_cat(FLAGS.cat)
   #print(cat)
   train, target = data_util.get_data(FLAGS.train, vocab, cat)
   test, test_target = data_util.get_data(FLAGS.test, vocab, cat)

   filter_size = map( int , FLAGS.filter_sizes.split(',') )
   num_filter = map( int , FLAGS.num_filters.split(',') )

   #print(filter_size)
   #print(num_filter)

   #print(train)
   #data.printsenfromid(train,vocab_re)
   #print(len(target[0]))
   model = rnn_model(vocab_size , FLAGS.batch_size, FLAGS.seq_max_len, FLAGS.lstm_size, len(target[0]), FLAGS.embedding_size , filter_size, num_filter, FLAGS.cell_type)

   init_op = tf.global_variables_initializer()
   saver = tf.train.Saver()
   with tf.Session() as sess:
      sess.run(init_op)
      in_test , in_test_target , in_test_seq_len = model.test_data(test, test_target)
      #print(np.array(in_test).shape)
      i = 0
      l = FLAGS.num_step
      print_progress(i, l, prefix = 'Train Progress:', suffix = 'Complete', bar_length = 50)
      for i in range(FLAGS.num_step):
         #print(i)
         #pass
         in_data, in_target, in_len = model.next_random_batch(train, target)
         #print(np.array(in_data).shape)

         #for z in in_data:
         #   print(np.array(model.get_embedding(sess,z)).shape)

         #print(np.array(model.step(sess, in_data, in_target, in_len, FLAGS.dropout)).shape)


         #model.step(sess, in_data , in_target, in_len , FLAGS.dropout)




         mmm = model.step(sess , in_data, in_target, in_len, FLAGS.dropout)
         i = i + 1
         print_progress(i, l, prefix = 'Train Progress:', suffix = 'Complete', bar_length = 50)
         
         #print(np.array(mmm).shape)
         #print(mmm[0])
         #for iii in mmm[0]:
         #   print(np.array(iii).shape)
         #print(np.array(mmm[2]).shape)
         #print(mmm[2])
         #pass
         #if i%1000 == 0:

            #print(model.test_acc( sess , in_test , in_test_target , in_test_seq_len , FLAGS.dropout))
      correct = 0
      total = 0
      print('runnint test set')
 
      for i,j,k in zip(in_test,in_test_target,in_test_seq_len):
         total = total +1
               #print([i],[j],[k],FLAGS.dropout)
               #print(model.test_acc(sess,[i],[j],[k],FLAGS.dropout))
         if model.test_acc(sess,[i],[j],[k],1.0)[0]:
                  #print('correct')
            correct = correct + 1
      print('accuracy on test set:'),
      print(correct/total)
      save_path = saver.save(sess,FLAGS.model+os.path.sep+'model.ckpt')
      
         
            #pass
            #print(np.array(in_test).shape)

         
         #print(in_target)
         #print(in_len)
         #val = model.step(sess, in_data, in_target, in_len)
         #print(str(i)+'\t'+str(val[0])+'\t'+str(val[1])),
         #print('\t'),
         #print(val[0]),
         #print('\t'),
         #print(val[1])
      #in_test , in_test_target, in_test_seq_len = model.test_data(test, test_target)
      #print(in_test)
   
      #correct = 0
      #total = 0
      #for i,j,k in zip(in_test, in_test_target , in_test_seq_len):
      #   total = total + 1
      #   if model.test_acc(sess,[i],[j],[k])


if __name__ == '__main__':
   tf.app.run()
