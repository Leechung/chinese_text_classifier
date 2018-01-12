
import os
import numpy as np
import tensorflow as tf

from train_chinese_lstm_cnn_embedding import rnn_model
import data as data_util
from config import FLAGS


def test():
   vocab, vocab_re = data_util.init_vocab(FLAGS.vocab)
   vocab_size = len(vocab)
   cat, cat_re = data_util.init_cat(FLAGS.cat)
   test, test_target = data_util.get_data(FLAGS.test, vocab, cat)

   filter_size = map( int , FLAGS.filter_sizes.split(',') )
   num_filter = map( int , FLAGS.num_filters.split(',') )

   model = rnn_model(vocab_size , FLAGS.batch_size, FLAGS.seq_max_len, FLAGS.lstm_size, len(test_target[0]), FLAGS.embedding_size , filter_size, num_filter, FLAGS.cell_type)

   saver = tf.train.Saver()
   with tf.Session() as sess:
      saver.restore(sess, FLAGS.model+os.path.sep+'model.ckpt')
      in_test , in_test_target , in_test_seq_len = model.test_data(test, test_target)
      correct = 0
      total = 0
      for i,j,k in zip(in_test, in_test_target, in_test_seq_len):
         total = total + 1
         #print(model.test_acc(sess,[i],[j],[k],1.0)[0])
         if model.test_acc(sess,[i],[j],[k],1.0)[0]:
            #print(correct)
            #print(total)
            correct = correct + 1
      print('accuracy on test set:')
      print(correct)
      print(total)
      print(float(correct)/total)


if __name__ == '__main__':
   test()
