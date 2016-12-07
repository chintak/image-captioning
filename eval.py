"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import
from __future__ import division

import os
from os.path import join, exists, isfile, isdir
import sys
sys.path.append('../')
sys.path.append('external/')
sys.path.append('../external/')
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat
import cPickle as pickle
from data_loader import load_dataset
from collections import defaultdict

from utils import CONFIG
from score import get_score

config = CONFIG.Trainer

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("model_path", "",
                       "Path to model for inference or eval mode.")
tf.flags.DEFINE_string("out_caption_path", "",
                       "Path to output the predicted captions.")
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

def decode_samples_to_captions(samples, id_to_word):
  captions = []
  for sample in samples:
    cap = []
    for idx in list(sample):
      w = id_to_word.get(idx, None)
      if w is None:
        continue
      elif w == '<ET>':
        cap.append('.')
        break
      cap.append(w)
    captions.append(' '.join(cap))
  return captions


def main(_, conf={}):
  assert FLAGS.model_path and exists(FLAGS.model_path)
  assert FLAGS.out_caption_path

  logger = config.log.getLogger(flag=1, name="root")
  logger.setLevel(logging.INFO)
  # print the experiment flags for logging purpose
  logger.info("python %s", stringify(sys.argv, ' '))

  ###########################################################################
  # new way of loading dataset
  ###########################################################################
  dataset_dir = 'data/%s' % FLAGS.dataset_name
  logger.info("Dataset path: %s", dataset_dir)
  ret = load_dataset(dataset_dir, split='train')
  (_, _, _, _, word_to_ids) = ret

  ret = load_dataset(dataset_dir, word_to_id=word_to_ids, split='val')
  (dev_raw_captions, dev_image_ids, dev_cnn_features, dev_img_to_idx, _) = ret

  id_to_word = dict([(v, k) for k, v in word_to_ids.iteritems()])

  dev_imgs = list(set(dev_image_ids))
  num_samples = len(dev_imgs)

  ###########################################################################
  # setup Session
  ###########################################################################
  sess = tf.Session()
  ###########################################################################
  # restore previously saved model to predict
  ###########################################################################
  model_path = FLAGS.model_path
  saver = tf.train.import_meta_graph('%s.meta' % model_path)
  saver.restore(sess, model_path)

  graph = tf.get_default_graph()
  images_t = graph.get_tensor_by_name('image_feature:0')
  generated_captions = graph.get_tensor_by_name('cap_generated:0')
  batch_size = int(generated_captions.get_shape()[0] or 1)
  graph.finalize()

  logger.info('Num of dev samples: {}'.format(num_samples))
  logger.info('Batch size: {}'.format(batch_size))

  # prepare ground truth captions
  gt_img_to_caps = defaultdict(list)
  for im, cp in zip(dev_image_ids, dev_raw_captions):
    gt_img_to_caps[im].append(cp)

  ###########################################################################
  # start predicting
  ###########################################################################
  hypo = {}
  ref = {}
  names = {}
  for start, end in zip(range(0, num_samples, batch_size),
                        range(batch_size, num_samples, batch_size)):
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

    logger.info('Processed %d/%d samples.', start, num_samples)

  final_scores = get_score(ref, hypo)

  logger.info('Bleu_1:\t%.3f', final_scores['Bleu_1'])
  logger.info('Bleu_2:\t%.3f', final_scores['Bleu_2'])
  logger.info('Bleu_3:\t%.3f', final_scores['Bleu_3'])
  logger.info('Bleu_4:\t%.3f', final_scores['Bleu_4'])
  logger.info('METEOR:\t%.3f', final_scores['METEOR'])
  logger.info('ROUGE_L:\t%.3f', final_scores['ROUGE_L'])
  logger.info('CIDEr:\t%.3f', final_scores['CIDEr'])

  with open(FLAGS.out_caption_path, 'w') as fp:
    for k in hypo.keys():
      fp.write('{} -> {}\n'.format(names[k], pformat(hypo[k])))
      fp.write('{} -> {}\n'.format(names[k], pformat(ref[k])))

  sess.close()


if __name__ == '__main__':
    tf.app.run()
