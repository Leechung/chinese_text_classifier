
from config import FLAGS
from collections import Counter
import operator

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def tokenize(sen):
   return sen.split()


def create_vocab(train_, vocab_, cat_, vocab_size = 50000):
   vocab = dict()
   cat = dict()
   with open(train_,'r') as f:
      for l in f:
         l = l.strip('\r').strip('\n')
         t = l.split('\t')
         if len(t) == 3:
            c = t[0]
            t = t[2]
            #print(tokenize(t))
            for i in tokenize(t):
               if i in vocab:
                  vocab[i] += 1
               else:
                  vocab[i] = 1
            if c in cat:
               cat[c] += 1
            else:
               cat[c] = 1
   #for i in vocab:
   #   print(i.decode('utf8'))
   vocab_uni = dict()
   for i in vocab:
      try:
         vocab_uni[unicode(i.decode('utf8'))] = vocab[i]
      except:
         print(i)
         #pass
   #print(vocab_uni)
   vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
   cat_list = sorted(cat , key=cat.get, reverse=True)
   if len(vocab_list) > vocab_size:
      vocab_list = vocab_list[:vocab_size]
   f = open(vocab_,'w')
   for i in vocab_list:
      f.write(i+'\n')
   f.close
   ff = open(cat_,'w')
   for i in cat_list:
      ff.write(i+'\n')
   ff.close()

def init_vocab(vocab_path):
   #pass
   re_vocab = []
   with open(vocab_path) as f:
      re_vocab.extend([ unicode(x.decode('utf8')) for x in  f.read().splitlines()])
   vocab = dict([(y,x) for (x,y) in enumerate(re_vocab)])
   return vocab, re_vocab

def init_cat(cat_path):
   re_cat = []
   with open(cat_path) as f:
      re_cat.extend(f.read().splitlines())
   cat = dict([(y,x) for (x,y) in enumerate(re_cat)])
   return cat, re_cat

def sentence2tokenid(sen, vocab):
   return [vocab.get(unicode(w.decode('utf8')),UNK_ID) for w in tokenize(sen)]

def get_data(data_path, vocab, cat):
   train = []
   target = []
   with open(data_path) as f:
      for l in f:
         l = l.strip('\n').strip('\r')
         a = l.split('\t')
         if len(a) is 3:
            t = [0]*len(cat)
            t[cat[a[0]]] = 1
            target.append(t)
            train.append(sentence2tokenid(a[2],vocab))
   return train, target


def id2sen(id_seq, vocab_re):
   return [vocab_re[x] for x in id_seq]

def printsenfromid(dat, vocab_re):
   for i in dat:
      print ' '.join(id2sen(i, vocab_re)) 

def main():
   create_vocab(FLAGS.train, FLAGS.vocab ,FLAGS.cat)
   vocab , vocab_re = init_vocab(FLAGS.vocab)
   cat, cat_re = init_cat(FLAGS.cat)
   #print cat
   train, target = get_data(FLAGS.test, vocab, cat)
   #print(target)
   printsenfromid(train, vocab_re)



if __name__ == '__main__':
   main()
