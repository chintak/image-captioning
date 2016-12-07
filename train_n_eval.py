"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import
from __future__ import division

import os
from os.path import join, exists, isfile, isdir
import sys
sys.path.append('../')
sys.path.append('external/')
sys.path.append('../external/')
import math
import shutil
import tensorflow as tf
import numpy as np
from pprint import pformat
import cPickle as pickle
import importlib
from collections import defaultdict

from data_loader import load_dataset
from im_cap_model import ImCapModel
from utils import CONFIG
from score import get_score

config = CONFIG.Trainer
logger = None


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("model_config_name", "",
                       "Which model config to use.")
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
tf.flags.DEFINE_string("dataset_name", "flickr8k",
                       "flickr8k, flickr30k or coco")


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


def restore_model(sess, path):
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
    restorer = tf.train.Saver()
    restorer.restore(sess, restore_path)
    logger.info('Restoring model from %s', restore_path)

def print_msg(*args):
  return "epoch: {} niter: {} batch_loss: {} curr_epoch_loss: {}".format(*args)

def decode_samples_to_captions(samples, id_to_word):
  captions = []
  for sample in samples:
    cap = []
    for idx in sample:
      w = id_to_word.get(idx, None)
      if w is None:
        continue
      elif w == '<ET>':
        break
      cap.append(w)
    captions.append(' '.join(cap))
  return captions


def main(_, conf={}):
  global logger
  try:
    mymodel = importlib.import_module(FLAGS.model_config_name)
  except:
    raise AttributeError("No model named %s" % FLAGS.model_config_name)
  solver_config = mymodel.solver
  solver_config.log_fname = solver_config.log_fname.replace(
      "expts", "expts/{}".format(FLAGS.dataset_name))
  logger = config.log.getLogger(flag=3, fname=solver_config.log_fname)
  # print the experiment flags for logging purpose
  logger.info("python %s", stringify(sys.argv, ' '))

  ###########################################################################
  # load train image captions
  ###########################################################################
  # assert exists(FLAGS.cnn_features_path)
  # assert exists(FLAGS.raw_captions_dir)
  # cnn_feats_path = FLAGS.cnn_features_path
  # raw_captions_dir = FLAGS.raw_captions_dir
  # train_cap_path = join(raw_captions_dir, 'Flickr8k.train.annotation.kl')
  # train_image_ids, train_raw_captions = unpickle(train_cap_path)

  # ###########################################################################
  # # load dev image captions
  # ###########################################################################
  # dev_cap_path = join(raw_captions_dir, 'Flickr8k.dev.annotation.kl')
  # dev_image_ids, dev_raw_captions = unpickle(dev_cap_path)

  # ###########################################################################
  # # load vocab
  # ###########################################################################
  # vocab_path = join(raw_captions_dir, 'vocab.kl')
  # word_to_ids = unpickle(vocab_path)
  # id_to_word = dict([(v, k) for k, v in word_to_ids.iteritems()])

  # ###########################################################################
  # # load cnn features
  # ###########################################################################
  # (train_img_to_idx, train_cnn_features) = load_cnn_features(cnn_feats_path)
  # dev_img_to_idx = train_img_to_idx

  ###########################################################################
  # new way of loading dataset
  ###########################################################################
  dataset_dir = 'data/%s' % FLAGS.dataset_name
  logger.info("Dataset path: %s", dataset_dir)
  ret = load_dataset(dataset_dir, split='train')
  (train_raw_captions, train_image_ids,
    train_cnn_features, train_img_to_idx, word_to_ids) = ret

  ret = load_dataset(dataset_dir, word_to_id=word_to_ids, split='val')
  (dev_raw_captions, dev_image_ids, dev_cnn_features, dev_img_to_idx, _) = ret

  id_to_word = dict([(v, k) for k, v in word_to_ids.iteritems()])

  ###########################################################################
  # eval related
  ###########################################################################
  dev_imgs = list(set(dev_image_ids))
  dev_num_samples = len(dev_imgs)
  # prepare ground truth captions
  gt_img_to_caps = defaultdict(list)
  for im, cp in zip(dev_image_ids, dev_raw_captions):
    gt_img_to_caps[im].append(cp)

  ###########################################################################
  # load the model config
  ###########################################################################
  model_config = mymodel.model
  num_samples = model_config.num_samples = len(train_image_ids)
  model_config.vocab_size = len(word_to_ids.keys()) + 1
  model_config.log_fname = model_config.log_fname.replace(
      "expts", "expts/{}".format(FLAGS.dataset_name))
  logger.info('Solver configuration: %s', pformat(solver_config))

  batch_size = model_config.batch_size
  num_epochs = solver_config.num_epochs
  solver_config.save_model_dir = solver_config.save_model_dir.replace(
      "expts", "expts/{}".format(FLAGS.dataset_name))
  if FLAGS.save_model_dir:
    save_model_dir = FLAGS.save_model_dir
    solver_config.save_model_dir = save_model_dir
  else:
    save_model_dir = solver_config.save_model_dir
  bool_save_model = True if save_model_dir else False
  if bool_save_model and not exists(save_model_dir):
    os.makedirs(save_model_dir)

  ###########################################################################
  # create the model
  ###########################################################################
  model = ImCapModel(model_config, word_to_ids)
  loss = model.build_model()
  tf.get_variable_scope().reuse_variables()
  generated_captions = model.build_generator()

  ###########################################################################
  # training related variables
  ###########################################################################
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

  ###########################################################################
  # setup saver
  ###########################################################################
  saver = tf.train.Saver(max_to_keep=solver_config.max_to_keep)

  ###########################################################################
  # setup Session and begin training
  ###########################################################################
  sess = tf.Session()
  coord = tf.train.Coordinator()

  init = tf.initialize_all_variables()
  sess.run(init)

  ###########################################################################
  # restore previously saved model to resume training
  ###########################################################################
  bool_resume_path = True if model_config.resume_from_model_path else False
  if bool_resume_path and exists(model_config.resume_from_model_path):
    restore_model(sess, model_config.resume_from_model_path)

  tf.get_default_graph().finalize()

  # compute the number of iterations reqd for training
  niters_per_epoch = num_samples // batch_size + 1
  num_iters_to_run = num_epochs * niters_per_epoch

  # placeholders
  image_feature = model_config.img_input_feed
  caption_feature = model_config.cap_input_feed

  # setup 10 random dev set images to check the generated captions
  num_gen_samples = model_config.batch_size
  samp_idx = np.random.randint(0, dev_raw_captions.shape[0], num_gen_samples)
  samp_captions = dev_raw_captions[samp_idx, :]
  samp_img_ids = dev_image_ids[samp_idx]
  im_ids = [dev_img_to_idx[ind] for ind in samp_img_ids]
  samp_cnn = train_cnn_features[im_ids, ...]
  logger.info('Sampling captions for: [%s]', ','.join(samp_img_ids[:10]))

  if bool_save_model:
    logger.info('Model save path: {}'.format(save_model_dir))
    model_save_freq = solver_config.ckpt_epoch_freq
    logger.info('Save model per iters: {}'.format(model_save_freq))
  logger.info('Initial learning rate: {}'.format(solver_config.learning_rate))
  logger.info('Num of samples: {}'.format(num_samples))
  logger.info('Num of iters: {}'.format(num_iters_to_run))

  ###########################################################################
  # start training
  ###########################################################################
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
      im_ids = [train_img_to_idx[ind] for ind in batch_img_ids]
      batch_cnn = train_cnn_features[im_ids, ...]

      feeder = {model.images: batch_cnn, model.input_seqs: batch_caps}
      _, batch_loss, niters = sess.run(
        [train_op, loss, global_step], feed_dict=feeder)
      epoch_loss += batch_loss

      if niters % 20 == 0:
        logger.info(print_msg(i, niters, batch_loss, epoch_loss))

      if start == 0:
        # generate some sample captions
        gen_samples = sess.run(generated_captions,
            feed_dict={model.images: samp_cnn})
        sampled_captions = decode_samples_to_captions(gen_samples, id_to_word)
        for j, (idx, cap) in enumerate(zip(samp_img_ids, sampled_captions)):
          logger.info('Generated caption: epoch %d, %s - %s',
                      i, idx, cap)
          if j == 10:
            break
        predict(sess, generated_captions, dev_cnn_features, dev_img_to_idx,
                dev_imgs, id_to_word, gt_img_to_caps, dev_num_samples,
                batch_size, model.images)

    if bool_save_model and i % (model_save_freq) == 0:
      # completed epoch, save model snapshot
      _path = saver.save(sess, solver_config.save_model_dir, global_step=i)
  if bool_save_model:
    # save the final model
    saver.save(sess, solver_config.save_model_dir, global_step=num_epochs)

  coord.request_stop()
  sess.close()


def predict(sess, generated_captions, dev_cnn_features, dev_img_to_idx,
            dev_imgs, id_to_word, gt_img_to_caps,
            num_samples, batch_size, images_t):
  global logger
  hypo = {}
  ref = {}
  names = {}
  for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    batch_imgs = dev_imgs[start:end]
    batch_ids = [dev_img_to_idx[k] for k in batch_imgs]
    batch_cnn = dev_cnn_features[batch_ids, ...]

    if batch_cnn.shape[0] < batch_size:
      batch_cnn = np.vstack(
          [batch_cnn, np.zeros((batch_size - batch_cnn.shape[0], 4096))])
    assert batch_cnn.shape[0] == batch_size

    feeder = {images_t: batch_cnn}
    gen_samples = sess.run(
      generated_captions, feed_dict=feeder)

    sampled_captions = decode_samples_to_captions(gen_samples, id_to_word)
    for i, name in enumerate(batch_imgs):
      hyp_cap = sampled_captions[i]
      caps = gt_img_to_caps[name]
      hypo[name] = [hyp_cap]
      names[name] = name
      ref[name] = decode_samples_to_captions(caps, id_to_word)

    print '\tProcessed %d/%d samples.' % (start, num_samples)

  assert len(ref.keys()) == len(hypo.keys())
  assert len(ref.keys()) == num_samples
  final_scores = get_score(ref, hypo)

  logger.info('\tBleu_1:\t%.3f', final_scores['Bleu_1'])
  logger.info('\tBleu_2:\t%.3f', final_scores['Bleu_2'])
  logger.info('\tBleu_3:\t%.3f', final_scores['Bleu_3'])
  logger.info('\tBleu_4:\t%.3f', final_scores['Bleu_4'])
  logger.info('\tMETEOR:\t%.3f', final_scores['METEOR'])
  logger.info('\tROUGE_L:\t%.3f', final_scores['ROUGE_L'])
  logger.info('\tCIDEr:\t%.3f', final_scores['CIDEr'])


if __name__ == '__main__':
    tf.app.run()
