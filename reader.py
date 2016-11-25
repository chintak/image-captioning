import os
import logging
import numpy as np
from utils import CONFIG
from pandas import read_csv
from pprint import pformat
from collections import Counter

config = CONFIG.CapData
logger = logging.getLogger('CapData')
logger.setLevel(config.logLevel)

__all__ = ['flickr8k_raw_data']


def _read_words(data_path):
    # read the captions and build a test language model on this data
    df = read_csv(data_path, delimiter='\t', names=['name', 'text'],
                  header=None)
    # return list of names and list of captions
    return (df.name.tolist(),
            [line.strip().split() for line in df.text.tolist()])


def _build_vocab(data_path):
    (_, cap_toks) = _read_words(data_path)
    words = [it for l in cap_toks for it in l]
    unq_words = map(lambda i: i[0], Counter(words).most_common()[::-1])
    vocab = dict(zip(unq_words, range(1, len(unq_words) + 1)))
    return vocab


def _file_to_ids(file_path, word_to_id):
    (names, cap_toks) = _read_words(file_path)
    return (names, [[word_to_id[word] for word in toks if word in word_to_id]
                    for toks in cap_toks])


def flickr8k_raw_data(data_path):
    train_path = os.path.join(data_path, 'Flickr8k.token.trainImgs.txt')
    test_path = os.path.join(data_path, 'Flickr8k.token.testImgs.txt')
    dev_path = os.path.join(data_path, 'Flickr8k.token.devImgs.txt')
    # vocab - dict, word -> int
    vocab = _build_vocab(train_path)
    # names - list, image name, word_to_ids - list, ints corresponding to word
    (tr_names, tr_word_to_ids) = _file_to_ids(train_path, vocab)
    (te_names, te_word_to_ids) = _file_to_ids(test_path, vocab)
    (de_names, de_word_to_ids) = _file_to_ids(dev_path, vocab)
    return ({'names': tr_names, 'word_to_ids': tr_word_to_ids},
            {'names': te_names, 'word_to_ids': te_word_to_ids},
            {'names': de_names, 'word_to_ids': de_word_to_ids},
            len(vocab.keys()))


if __name__ == '__main__':
    fname = '../Flickr8k_Captions/'
    # fname = '../simple-examples/data/ptb.train.txt'

