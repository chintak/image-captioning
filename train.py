from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import configuration
from utils import CONFIG

config = CONFIG.Model
logger = logging.getLogger("Model")
logger.setLevel(config.logLevel)

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("image_features_path", "",
                       "Path to the VGG features of all the images.")
tf.flags.DEFINE_string("image_model_path", "",
                       "Path to VGG/ResNet model for feature extraction.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 90000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")


def main(_):
    assert (FLAGS.image_features_path or FLAGS.image_model_path), (
        "--image_features_path or --image_model_path expected.")
    assert FLAGS.train_dir, "--train_dir expected"

    # load the model and training config
    model_config = configuration.CapConfig(
        image_features_path=FLAGS.image_features_path)
    train_config = configuration.TrainConfig()

    # set up the directory for saving model checkpoints
    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        logger.info("Saving model checkpoints in %s" % FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # set up tensorflow graph for building the model
    g = tf.Graph()
    with g.as_default():



if __name__ == '__main__':
    tf.app.run()

