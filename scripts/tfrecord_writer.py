import os
import argparse
import tensorflow as tf
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('cnn_feats_path', help='a numpy.mmap expected')
    parser.add_argument(
        'caption_tokens_dir',
        help='Directory containing train, test and dev captions.')
    args = parser.parse_args()
    return args


def main():
    args = arguments()
    # read the mmap file containing CNN features
    feats_fname = os.path.splitext(os.path.basename(args.cnn_feats_path))[0]
    feats_shape = tuple([int(i) for i in feats_fname.split('_')[-1].split('X')])
    feats_mmap = np.memmap(args.cnn_feats_path, mode='r',  # read-only
                           shape=feats_shape, dtype=np.float32)

