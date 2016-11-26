from __future__ import print_function
import os
import logging
import numpy as np
from synset import *
from dataset_loader import DatasetLoader
from utils import CONFIG

config = CONFIG.ResNetFeatureExtractor
log = logging.getLogger('FeatureExtractor')
log.setLevel(config.log.level)


def print_prob(prob):
    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    log.info("Top1: ", top1)
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    log.debug("Top5: ", top5)
    return top1


def resnet_extract_features(data_loader, meta_path, ckpt_path):
    import tensorflow as tf

    save_path = "{}_feats_resnet{}_{}X{}.mnp".format(
        os.path.abspath(data_loader.folder_name), data_loader.input_size,
        data_loader.num_samples, config.feat_size)
    if os.path.exists(save_path):
        log.fatal("path {} present, no overwriting".format(save_path))
        return
    log.info("features saved in: {}".format(save_path))
    feats_mmap = np.memmap(save_path, mode='w+',
                           shape=(data_loader.num_samples, config.feat_size),
                           dtype=np.float32)
    with tf.device('/gpu:0'):
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, ckpt_path)

        graph = tf.get_default_graph()
        feat_tensor = graph.get_tensor_by_name(config.output_layer)
        images = graph.get_tensor_by_name(config.input_layer)
        # for op in graph.get_operations():
        #     print(op.name)
        #init = tf.initialize_all_variables()
        # sess.run(init)
        log.info("graph restored")
        log.info("starting feature extraction")

        idx = 0
        while data_loader.passes == 0:
            batch = data_loader.get_batch(config.batch_size, warp=False)
            batch_size = batch.shape[0]
            log.debug("forward prop batch of size: {}".format(batch_size))
            feed_dict = {images: batch}
            feats = sess.run(feat_tensor, feed_dict=feed_dict)
            log.debug("size of output feat: {}".format(feats.shape))
            feats_mmap[idx:idx + batch_size, :] = feats[:batch_size, :]
            idx = idx + batch_size
            log.info("Processed batch {}/{} images".format(
                idx, data_loader.num_samples))

        sess.close()
        tf.reset_default_graph()
    log.info("Processing completed.")


def arguments():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--folder-path", help="path to folder containing images")
    args.add_argument("--ckpt-path",
                      help="resnet checkpoint path (.ckpt)")
    args.add_argument("--meta-path", help="resnet meta path (.meta)")
    parser = args.parse_args()
    assert (os.path.exists(parser.ckpt_path) and
            os.path.exists(parser.meta_path)), (
        "Valid paths to ResNet model not provided.")
    if not os.path.isdir(parser.folder_path):
        log.fatal("Folder expected")
        import sys
        sys.exit(0)
    return parser

if __name__ == '__main__':
    args = arguments()
    data_loader = DatasetLoader(args.folder_path)
    resnet_extract_features(data_loader, args.meta_path, args.ckpt_path)
