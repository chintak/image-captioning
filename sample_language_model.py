"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import

import os
import argparse
import tensorflow as tf

from reader import flickr8k_raw_data
from configuration import CapConfig


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    args = parser.parse_args()
    return args


def main():
    args = arguments()
    train, test, eval, vocab_len = flickr8k_raw_data(args.data_path)

    g = tf.Graph()
    with g.as_default():



if __name__ == '__main__':
    main()

