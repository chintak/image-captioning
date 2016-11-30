from __future__ import print_function
import os
import logging
import numpy as np
from synset import *
from dataset_loader import DatasetLoader
from utils import CONFIG

config = CONFIG.VGGFeatureExtractor
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


def vgg_extract_features(data_loader, model_path):
    import tensorflow as tf

    save_path = "{}_feats_vgg{}_{}X{}.mnp".format(
        os.path.abspath(data_loader.folder_name), data_loader.input_size,
        data_loader.num_samples, config.feat_size)
    if os.path.exists(save_path):
        log.fatal("path {} present, no overwriting".format(save_path))
        return
    feats_mmap = np.memmap(save_path, mode='w+',
                           shape=(data_loader.num_samples, config.feat_size),
                           dtype=np.float32)
    with open(model_path, mode='rb') as f:
        fileContent = f.read()

    with tf.device('/gpu:0'):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)
        images = tf.placeholder("float", [None, 224, 224, 3])
        tf.import_graph_def(graph_def, input_map={"images": images})
        log.info("graph loaded from disk")

        graph = tf.get_default_graph()

        with tf.Session() as sess:
            feat_tensor = graph.get_tensor_by_name(config.output_layer)
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

        tf.reset_default_graph()
    log.info("Processing completed.")


def arguments():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--folder-path", help="path to folder containing images")
    args.add_argument("--model-path",
                      help="vgg model path (.tfmodel)")
    parser = args.parse_args()
    assert os.path.exists(parser.model_path), (
        "Valid paths to VGG model not provided.")
    if not os.path.isdir(parser.folder_path):
        log.fatal("Folder expected")
        import sys
        sys.exit(0)
    return parser

if __name__ == '__main__':
    args = arguments()
    data_loader = DatasetLoader(args.folder_path)
    vgg_extract_features(data_loader, args.model_path)
