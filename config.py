import tensorflow as tf



tf.app.flags.DEFINE_string('train','../data/sougou/train','chinese training data.')
tf.app.flags.DEFINE_string('test','../data/sougou/test', 'chinese testing data.')
tf.app.flags.DEFINE_string('vocab','../data/sougou/vocab','save vocabulary here.')
tf.app.flags.DEFINE_string('cat','../data/sougou/cat','save categories here.')
tf.app.flags.DEFINE_string('model','../model/sougou','save model here.')

tf.app.flags.DEFINE_integer('vocab_max_size', 10000 , 'size of character.')
tf.app.flags.DEFINE_integer('batch_size', 64 , 'size of training batch.')
tf.app.flags.DEFINE_integer('embedding_size', 64, 'embedding size of character.')

tf.app.flags.DEFINE_float('learning_rate' , 0.01, 'learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor' , 0.9999, 'learning rate decay factor.')

tf.app.flags.DEFINE_integer('num_step', 100000 , 'number of learning steps/iterations.')
tf.app.flags.DEFINE_integer('seq_max_len', 1000, 'maximum length of sequence.')
tf.app.flags.DEFINE_integer('lstm_size',128,'size of lstm cell')

# CNN parameters
tf.app.flags.DEFINE_string('filter_sizes','1,2,3,4,5,6,7,8','comma seperated filter sizes.')
tf.app.flags.DEFINE_string('num_filters','200,400,400,300,300,200,200,200','number of filters per filter size.')
tf.app.flags.DEFINE_float('dropout',0.5,'dropout keep probability.')
tf.app.flags.DEFINE_string('cell_type','lstm','cell type empty for gru lstm')

FLAGS = tf.app.flags.FLAGS
