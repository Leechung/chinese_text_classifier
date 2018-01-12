
import data as data_util
from config import FLAGS

data_util.create_vocab(FLAGS.train, FLAGS.vocab, FLAGS.cat, vocab_size = FLAGS.vocab_max_size)
vocab, vocab_re = data_util.init_vocab(FLAGS.vocab)
vocab_size = len(vocab)
cat, cat_re = data_util.init_cat(FLAGS.cat)
#print(cat)
train, target = data_util.get_data(FLAGS.train, vocab, cat)
test, test_target = data_util.get_data(FLAGS.test, vocab, cat)

wc = 0

minn = 0
maxn = 0


for i in train:
   if len(i) < 500:
      minn=minn+1
   wc = wc+len(i)
   #if len(i) < minn:
   #   minn = len(i)
   #if len(i) > maxn:
   #   maxn = len(i)

for i in test:
   if len(i) > 10000:
      maxn=maxn+1
   wc = wc+len(i)
   #if len(i) < minn:
   #   minn = len(i)
   #if len(i) > maxn:
   #   maxn = len(i)

print(wc)
print(len(train)+len(test))
print(minn)
print(maxn)
