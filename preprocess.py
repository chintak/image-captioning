import os
import logging
import numpy as np
from utils import CONFIG
from pandas import read_csv
from pprint import pformat
from collections import Counter
try:
  # import hickle as pickle
  import cPickle as pickle
except:
  import pickle
try:
  from nltk.tokenize import word_tokenize
except:
  word_tokenize = lambda t: t.split(" ")


config = CONFIG.CapData
logger = logging.getLogger('CapData')
logger.setLevel(config.log.level)

MAX_LENGTH = config.max_length
assert MAX_LENGTH
MIN_COUNT = config.min_freq
assert MIN_COUNT

# __all__ = ['flickr8k_raw_data']

def _parse_caps(txt):
  txt = txt.strip().lower().replace(",", "").replace("'", "")
  txt = txt.replace(".", "").replace("\"", "").replace("!", "")
  txt = txt.replace("?", "").replace("-", "").replace(')', '')
  txt = txt.replace("(", "").replace("&", "and")
  txt = "{} .".format(" ".join(txt.split()).strip())
  return txt


def read_caps(data_path):
  # read the captions and build a test language model on this data
  df = read_csv(data_path, delimiter='\t', names=['name', 'text'], header=None)
  df['image_id'] = map(lambda k: k.split('#')[0], df.name)
  df.text = map(_parse_caps, df.text)
  df['len'] = map(lambda k: len(k.split(' ')), df.text)
  df = df.ix[df['len'] < MAX_LENGTH]
  del df['len']
  del df['name']
  return (df.image_id.tolist(), df.text.tolist())


def _strip_name(paths):
  return [(os.path.basename(p), i) for i, p in enumerate(paths)]


def main():
  args = arguments()
  # read the mmap file containing CNN features
  feats_fname = os.path.splitext(os.path.basename(args.cnn_feats_path))[0]
  img_name_list_path = os.path.join(
      os.path.dirname(args.cnn_feats_path),
      '{}_list.txt'.format(
          '_'.join(feats_fname.split('_')[:-3])))
  feats_shape = tuple([int(i) for i in feats_fname.split('_')[-1].split('X')])
  feats_mmap = np.memmap(args.cnn_feats_path, mode='r',  # read-only
                         shape=feats_shape, dtype=np.float32)
  img_to_idx = {}
  with open(img_name_list_path, 'r') as fp:
    img_to_idx = dict(_strip_name(fp.read().split('\n')))


def build_vocab(data_path):
  (_, captions) = read_caps(data_path)
  words = [word for line in captions for word in word_tokenize(line)]
  unq_words = map(lambda i: i[0], Counter(words).most_common()[::-1])
  vocab = {'<PAD>': 0, '<ST>': 1, '<ET>': 2}
  idx = 3
  for w in unq_words:
    vocab[w] = idx
    idx += 1
  print "Max number of words:", idx
  return vocab


def _file_to_ids(file_path, word_to_id):
  (names, captions) = read_caps(file_path)
  word_ids_shape = (len(captions), MAX_LENGTH + 2)
  word_ids = word_to_id['<PAD>'] * np.ones(word_ids_shape, dtype=np.int64)

  for i, line in enumerate(captions):
    word_ids[i, 0] = word_to_id['<ST>']
    idx = 1
    for j, word in enumerate(word_tokenize(line)):
      w2i = word_to_id.get(word, None)
      if w2i:
        word_ids[i, idx] = w2i
      idx += 1
    word_ids[i, idx] = word_to_id['<ET>']
  names = np.asarray(names)
  return (names, word_ids)


def save_pick(path, val):
  with open(path, 'w') as fp:
    pickle.dump(val, fp, protocol=2)


def flickr8k_raw_data(data_path):
  train_path = os.path.join(data_path, 'Flickr8k.token.trainImgs.txt')
  test_path = os.path.join(data_path, 'Flickr8k.token.testImgs.txt')
  dev_path = os.path.join(data_path, 'Flickr8k.token.devImgs.txt')
  # vocab - dict, word -> int
  vocab = build_vocab(train_path)
  print 'Vocabulary built.'
  pkl_vocab_path = os.path.join(data_path, 'vocab.kl')
  save_pick(pkl_vocab_path, vocab)
  print 'Vocabulary saved to', pkl_vocab_path
  # names - list, image name, word_to_ids - list, ints corresponding to word
  for split in ['train', 'test', 'dev']:
    path = os.path.join(data_path, 'Flickr8k.token.%sImgs.txt' % split)
    save_path = os.path.join(data_path, 'Flickr8k.%s.annotation.kl' % split)
    (names, word_ids) = _file_to_ids(path, vocab)
    save_pick(save_path, (names, word_ids))
    print "Saved annotations for %s set." % split
  print 'Captions tokenized and encoded.'


if __name__ == '__main__':
  flickr8k_raw_data('../data/Flickr8k_Captions')

