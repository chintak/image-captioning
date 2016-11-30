"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import
from __future__ import division

import os
from os.path import join, exists, isfile, isdir
import sys
import math
import shutil
import tensorflow as tf
import numpy as np
from pprint import pformat
import cPickle as pickle
import importlib

from im_cap_model import ImCapModel
from utils import CONFIG

config = CONFIG.Trainer
logger = None


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("model_config_name", "",
                       "Which model config to use.")
tf.flags.DEFINE_string("cnn_features_path", "",
                       "Path to file containing cnn features.")
tf.flags.DEFINE_string("raw_captions_dir", "",
                       "Path to file containing raw captions.")
tf.flags.DEFINE_string("save_model_dir", "",
                       "Path to dir to save model checkpoint.")
tf.flags.DEFINE_string("model_path", "",
                       "Path to model for inference or eval mode.")
tf.flags.DEFINE_string("out_caption_path", "",
                       "Path to output the predicted captions.")
tf.flags.DEFINE_string("resume_from_model_path", None,
                       "Path to model checkpoint to resume training.")
tf.flags.DEFINE_string("mode", "train",
                       "Run mode.")


def unpickle(path):
  with open(path, 'rb') as fp:
    res = pickle.load(fp)
  return res

def _strip_name(paths):
  return [(os.path.basename(p), i) for i, p in enumerate(paths)]


def load_cnn_features(path):
  feats_dir = os.path.splitext(os.path.basename(path))[0]
  img_name_list_path = join(
      os.path.dirname(path),
      '{}_list.txt'.format('_'.join(feats_dir.split('_')[:-3])))
  feats_shape = tuple([int(i) for i in feats_dir.split('_')[-1].split('X')])
  feats_mmap = np.memmap(path, mode='r',  # read-only
                         shape=feats_shape, dtype=np.float32)
  img_to_idx = {}
  with open(img_name_list_path, 'r') as fp:
    img_to_idx = dict(_strip_name(fp.read().split('\n')))

  return (img_to_idx, feats_mmap)


def stringify(ls, sep=', '):
  return sep.join([str(i) for i in ls])


def restore_model(path):
  if isfile(path):
    restore_path = path
  else:
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and exists(ckpt.model_checkpoint_path):
      restore_path = ckpt.model_checkpoint_path
      # restored_step = int(ckpt.model_checkpoint_path.split('-')[-1])
    else:
      logger.fatal('No restorable checkpoint model found.')
  if restore_path:
    restorer = tf.train.Saver(tf.all_variables())
    restorer.restore(sess, restore_path)
    logger.info('Restoring model from %s', restore_path)

def print_msg(*args):
  return "epoch: {} niter: {} batch_loss: {} curr_epoch_loss: {}".format(*args)

def main(_, conf={}):
  global logger
  assert exists(FLAGS.cnn_features_path)
  assert exists(FLAGS.raw_captions_dir)
  cnn_feats_path = FLAGS.cnn_features_path
  raw_captions_dir = FLAGS.raw_captions_dir
  try:
    mymodel = importlib.import_module(FLAGS.model_config_name)
  except:
    raise AttributeError("No model named %s" % FLAGS.model_config_name)
  solver_config = mymodel.solver
  logger = config.log.getLogger(flag=3, fname=solver_config.log_fname)
  # print the experiment flags for logging purpose
  logger.info("python %s", stringify(sys.argv, ' '))

  # load image captions
  train_cap_path = join(raw_captions_dir, 'Flickr8k.train.annotation.kl')
  train_image_ids, train_raw_captions = unpickle(train_cap_path)

  # load vocab
  vocab_path = join(raw_captions_dir, 'vocab.kl')
  word_to_ids = unpickle(vocab_path)

  # load cnn features
  (img_to_idx, train_cnn_features) = load_cnn_features(cnn_feats_path)

  # load the model config
  model_config = mymodel.model
  num_samples = model_config.num_samples = len(train_image_ids)

  batch_size = model_config.batch_size
  num_epochs = solver_config.num_epochs
  save_model_dir = solver_config.get('save_model_dir', FLAGS.save_model_dir)
  bool_save_model = True if save_model_dir else False
  if bool_save_model and not exists(save_model_dir):
    os.makedirs(save_model_dir)

  # create the model
  model = ImCapModel(model_config, word_to_ids)
  loss = model.build_model()
  tf.get_variable_scope().reuse_variables()

  global_step = tf.Variable(
      initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

  learning_rate = tf.constant(solver_config.learning_rate)
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=global_step,
      learning_rate=learning_rate,
      optimizer=solver_config.optimizer,
      clip_gradients=solver_config.train_clip_gradients,
      learning_rate_decay_fn=None)

  # setup saver
  saver = tf.train.Saver(max_to_keep=solver_config.max_to_keep)

  # setup Session and begin training
  sess = tf.Session()
  coord = tf.train.Coordinator()

  init = tf.initialize_all_variables()
  sess.run(init)

  # restore previously saved model to resume training
  bool_resume_path = True if model_config.resume_from_model_path else False
  if bool_resume_path and exists(model_config.resume_from_model_path):
    restore_model(model_config.resume_from_model_path)

  tf.get_default_graph().finalize()

  # compute the number of iterations reqd for training
  niters_per_epoch = num_samples // batch_size + 1
  num_iters_to_run = num_epochs * niters_per_epoch

  # placeholders
  image_feature = model_config.img_input_feed
  caption_feature = model_config.cap_input_feed

  if bool_save_model:
    logger.info('Model save path: {}'.format(save_model_dir))
    model_save_freq = solver_config.ckpt_epoch_freq
    logger.info('Save model per iters: {}'.format(model_save_freq))
  logger.info('Initial learning rate: {}'.format(solver_config.learning_rate))
  logger.info('Num of samples: {}'.format(num_samples))
  logger.info('Num of iters: {}'.format(num_iters_to_run))

  for i in range(num_epochs):
    epoch_loss = 0.
    idxs = np.random.permutation(num_samples)
    # caption features: train_image_ids, train_raw_captions
    # image features: img_to_idx, train_cnn_features
    epoch_captions = train_raw_captions[idxs, :]
    epoch_image_ids = train_image_ids[idxs]

    for start, end in zip(range(0, num_samples, batch_size),
                          range(batch_size, num_samples, batch_size)):
      batch_caps = epoch_captions[start:end, :]
      batch_img_ids = epoch_image_ids[start:end]
      im_ids = [img_to_idx[ind] for ind in batch_img_ids]
      batch_cnn = train_cnn_features[im_ids, ...]

      feeder = {model.images: batch_cnn, model.input_seqs: batch_caps}
      _, batch_loss, niters = sess.run(
        [train_op, loss, global_step], feed_dict=feeder)
      epoch_loss += batch_loss

      if niters % 20 == 0:
        logger.info(print_msg(i, niters, batch_loss, epoch_loss))

    if bool_save_model and i % (model_save_freq) == 0:
      # completed epoch, save model snapshot
      _path = saver.save(sess, solver_config.save_model_dir, global_step=i)
  if bool_save_model:
    # save the final model
    saver.save(sess, solver_config.save_model_dir, global_step=niters)

  coord.request_stop()
  sess.close()


if __name__ == '__main__':
    tf.app.run()

