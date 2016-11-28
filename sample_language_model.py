"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import

import os
import sys
import math
import shutil
import tensorflow as tf
import numpy as np
from pprint import pformat

from reader import flickr8k_raw_data
from configuration import CapConfig
from external.inputs import batch_with_dynamic_pad
from utils import CONFIG

config = CONFIG.Trainer
logger = None


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("tfrecord_path", "",
                       "Path to training data in TFRecord format.")
tf.flags.DEFINE_string("save_model_dir", "",
                       "Path to dir to save model checkpoint.")
tf.flags.DEFINE_string("model_path", "",
                       "Path to model for inference or eval mode.")
tf.flags.DEFINE_string("out_caption_path", "",
                       "Path to output the predicted captions.")
tf.flags.DEFINE_integer("num_epochs", 10,
                        "Num of epochs for training the model.")
tf.flags.DEFINE_string("resume_from_model_path", None,
                       "Path to model checkpoint to resume training.")
tf.flags.DEFINE_string("lr_decay_method", None,
                       "'exp' or 'piecewise' learning rate decay schedule.")
tf.flags.DEFINE_string("lr_decay_boundaries", None,
                       "Boundaries for piecewise learning rate decay.")
tf.flags.DEFINE_string("lr_decay_values", "2.0,1.0,0.5,0.1,0.01",
                       "Values to use for decaying lr using piecewise decay.")
tf.flags.DEFINE_string("mode", "train",
                       "Run mode.")
tf.flags.DEFINE_string("optimizer", "SGD",
                       "Optimizer to use - SGD or Adam.")

def parse_seq_example(serialized, image_feature, caption_feature, id="id"):
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={
      image_feature: tf.FixedLenFeature([4096], dtype=tf.float32),
      id: tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
      caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64)
  })

  return context[image_feature], sequence[caption_feature], context[id]


def print_msg(*args):
  return "niter: {} lr: {} batch_loss: {} avg_batch_loss: {}".format(*args)


def str_to_real_list(sval, delim=','):
  return [float(v) for v in sval.split(delim)]


def str_to_int_list(sval, delim=','):
  return [int(v) for v in sval.split(delim)]


def stringify(ls, sep=', '):
  return sep.join([str(i) for i in ls])


def piecewise_constant(x, bounds, vals):
  assert bounds
  assert vals
  endb = len(bounds) - 1
  endv = len(vals) - 1
  bounds = tf.convert_to_tensor(bounds, dtype=tf.int32)
  vals = tf.convert_to_tensor(vals, dtype=tf.float32)
  pred_pair_fn = {}
  pred_pair_fn[x <= bounds[0]] = lambda: vals[0]
  pred_pair_fn[x > bounds[endb]] = lambda: vals[endv]
  # for low, high, v in zip(bounds[:endb], bounds[1:], vals[1:endv]):
  for i in range(endb):
    v = vals[i + 1]
    pred = (x > bounds[i]) & (x <= bounds[i + 1])
    pred_pair_fn[pred] = lambda v=v: v
    # pred_pair_fn[pred] = lambda v=v: v

  default = lambda: vals[0]
  return tf.case(pred_pair_fn, default, exclusive=True)


def main(_):
  global logger

  assert FLAGS.mode in ['train', 'eval', 'inference']
  if FLAGS.mode in ['eval', 'inference']:
    assert os.path.exists(FLAGS.model_path)
    assert FLAGS.out_caption_path
    model_path = (tf.train.latest_checkpoint(FLAGS.model_path)
                  if os.path.isdir(FLAGS.model_path) else FLAGS.model_path)
    log_fname = '{}-eval.log'.format(model_path)
  else:
    assert FLAGS.tfrecord_path
    assert FLAGS.lr_decay_method in ['exp', 'piecewise', None]
    assert FLAGS.save_model_dir
    if not os.path.exists(FLAGS.save_model_dir):
      os.makedirs(FLAGS.save_model_dir)
    save_model_path = os.path.join(FLAGS.save_model_dir, 'model')
    log_fname = '{}.log'.format(os.path.join(FLAGS.save_model_dir, 'run'))

  mode = FLAGS.mode

  logger = config.log.getLogger(
      flag=3 if mode == 'train' else 1, fname=log_fname)
  logger.setLevel(config.log.level)
  # print the experiment flags for logging purpose
  logger.info("python %s", stringify(sys.argv, ' '))

  # Config variables
  c = 0
  for record in tf.python_io.tf_record_iterator(FLAGS.tfrecord_path):
     c += 1
  num_train_samples = c
  num_epochs = FLAGS.num_epochs if FLAGS.mode == 'train' else 1
  batch_size = 128 if mode == 'train' else 1
  num_reader_threads = 4
  ckpt_epoch_freq = 5

  initializer_scale = 0.08
  initializer = tf.random_uniform_initializer(
      minval=-initializer_scale,
      maxval=initializer_scale)

  vocab_size = 10000
  embedding_size = 512
  num_lstm_units = 512
  lstm_dropout_prob = 0.7

  vals_queue_name = 'value_queue'
  image_feature = 'image_feature'
  caption_feature = 'caption_feature'

  train_clip_gradients = 5.0
  optimizer_name = FLAGS.optimizer
  if optimizer_name == 'SGD':
    initial_learning_rate = 2.0
  else:
    initial_learning_rate = .001
  lr_decay_method = FLAGS.lr_decay_method
  # use the following params for exponential lr decay
  learning_rate_decay_rate = 0.5
  num_epochs_per_decay = min(num_epochs // 5, 10)
  # use the following params for piecewise constant lr decay
  num_decay_steps = 5
  # FLAGS.lr_decay_values: [2.0 1.0 0.5 0.1]
  lr_decay_values = str_to_real_list(FLAGS.lr_decay_values, ',')

  # setup the inputs
  filename_queue = tf.train.string_input_producer([FLAGS.tfrecord_path])

  reader = tf.TFRecordReader()

  # set up input queue to prefetch tfrecords from the file
  values_queue = tf.FIFOQueue(
      capacity=10 * batch_size,
      dtypes=[tf.string],
      name="fifo_" + vals_queue_name
  )
  enqueue_ops = []
  for i in range(num_reader_threads):
    _, val = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([val]))
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
    values_queue, enqueue_ops))

  # extract batch of image and caption pairs
  images_and_captions = []
  for i in range(batch_size):
    serialized_sample = values_queue.dequeue()

    image, input_cap, image_id = parse_seq_example(
        serialized_sample, image_feature, caption_feature)
    images_and_captions.append([image, input_cap])

  # create dynamic padded batches for this captions and image features

  images, input_seqs, target_seqs, input_mask = batch_with_dynamic_pad(
      images_and_captions,
      batch_size=batch_size,
      queue_capacity=10 * batch_size)

  # create the model
  # - embedding layer for cnn features
  # - embedding layer for the sequence features
  # - lstm cell
  # - dynamic rnn using lstm cell

  # image embedding layer
  with tf.variable_scope("image_embedding") as scope:
    image_embeddings = tf.contrib.layers.fully_connected(
      images,
      num_output_units=embedding_size,
      activation_fn=None,
      weight_init=initializer,
      bias_init=None,
      name=scope)

  # sequence embedding layer
  with tf.variable_scope("seq_embedding"), tf.device('/cpu:0'):
    embedding_map = tf.get_variable(
        name="map",
        shape=[vocab_size, embedding_size],
        initializer=initializer)
    seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

  # image features are stored in image_embeddings with shape [batch_size,
  # embedding_size]
  # caption features are stored in seq_embeddings with shape [batch_size,
  # seq_length, embedding_size]

  # create lstm cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(
      num_units=num_lstm_units)
  if mode == 'train':
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=lstm_dropout_prob,
        output_keep_prob=lstm_dropout_prob)

  with tf.variable_scope('lstm', initializer=initializer) as lstm_scope:
    # initialize the lstm with image embeddings
    zero_state = cell.zero_state(
        batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
    _, initial_state = cell(image_embeddings, zero_state)

    lstm_scope.reuse_variables()

    if mode == 'inference':
      # TODO: add later
      assert False
      pass
    else:
      seq_lengths = tf.reduce_sum(input_mask, 1)
      lstm_outputs, _ = tf.nn.dynamic_rnn(
          cell=cell, inputs=seq_embeddings,
          sequence_length=seq_lengths, initial_state=initial_state,
          dtype=tf.float32, scope=lstm_scope)

  # stack batches vertically
  lstm_outputs = tf.reshape(lstm_outputs, [-1, cell.output_size])

  with tf.variable_scope('logits') as logits_scope:
    logits = tf.contrib.layers.fully_connected(
        lstm_outputs,
        num_output_units=vocab_size,
        activation_fn=None,
        weight_init=initializer,
        name=logits_scope)

  if mode == 'inference':
    tf.nn.softmax(logits, name='softmax')
  else:
    targets = tf.reshape(target_seqs, [-1])
    weights = tf.to_float(tf.reshape(input_mask, [-1]))

    # Compute losses.
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    batch_loss = tf.div(tf.reduce_sum(tf.mul(losses, weights)),
                        tf.reduce_sum(weights),
                        name="batch_loss")
    tf.add_to_collection("LOSSES", batch_loss)

    loss_ops = tf.get_collection("LOSSES")
    total_loss = tf.add_n(loss_ops, name="total_loss")

    # Add summaries.
    tf.scalar_summary("batch_loss", batch_loss)
    tf.scalar_summary("total_loss", total_loss)
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)

  # loss variables used for evaluation
  target_cross_entropy_losses = losses
  target_cross_entropy_loss_weights = weights

  global_step = tf.Variable(
      initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

  # Model setup complete

  # compute the number of iterations reqd for training
  num_batches_per_epoch = num_train_samples // batch_size + 1
  num_iters_to_run = num_epochs * num_batches_per_epoch
  num_epochs_done = 0
  mean_loss_acc = 0.

  # set up learing rate decay and learning rate decay
  learning_rate_decay_fn = None
  if optimizer_name.lower() == 'sgd' and lr_decay_method == 'piecewise':
    num_steps = len(lr_decay_values)
    if FLAGS.lr_decay_boundaries:
      lr_vs = str_to_int_list(FLAGS.lr_decay_boundaries, ',')
      lr_decay_boundaries = [num_batches_per_epoch * i for i in lr_vs]
    else:
      batches_per_decay = (num_batches_per_epoch * num_epochs) // num_steps
      lr_decay_boundaries = [batches_per_decay * i for i in range(1, num_steps)]
    learning_rate = piecewise_constant(
          global_step,
          lr_decay_boundaries,
          lr_decay_values)
    logger.info('Learning rate with piecewise constant decay %s - %s',
                stringify(lr_decay_boundaries), stringify(lr_decay_values))
  elif (optimizer_name.lower() == 'sgd' and lr_decay_method == 'exp' and
        learning_rate_decay_rate > 0):
    initial_decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    decay_steps = tf.Variable(
        initial_value=initial_decay_steps,
        name="decay_steps",
        trainable=False)
    learning_rate = tf.train.exponential_decay(
          initial_learning_rate,
          global_step,
          decay_steps,
          learning_rate_decay_rate,
          staircase=True)
    logger.info('Learning rate decay configured after every %s steps.',
                initial_decay_steps)
  else:
    learning_rate = tf.constant(initial_learning_rate)

  # now setup the optimizer for training the model
  train_op = tf.contrib.layers.optimize_loss(
      loss=batch_loss,
      global_step=global_step,
      learning_rate=learning_rate,
      optimizer=optimizer_name,
      clip_gradients=train_clip_gradients,
      learning_rate_decay_fn=learning_rate_decay_fn)

  # setup saver
  saver = tf.train.Saver(max_to_keep=0)

  # setup Session and begin training
  sess = tf.Session()
  coord = tf.train.Coordinator()

  init = tf.initialize_all_variables()
  sess.run(init)

  logger.info('TFRecord file: {}'.format(FLAGS.tfrecord_path))
  if mode == 'train':
    logger.info('Mode save path: {}'.format(FLAGS.save_model_dir))
    logger.info('Initial learning rate: {}'.format(initial_learning_rate))
  logger.info('Num of samples: {}'.format(num_train_samples))
  logger.info('Num of iters: {}'.format(num_iters_to_run))

  # restore previously saved model to resume training
  if FLAGS.resume_from_model_path:
    restore_path = None
    if os.path.isfile(FLAGS.resume_from_model_path):
      restore_path = FLAGS.resume_from_model_path
    else:
      ckpt = tf.train.get_checkpoint_state(FLAGS.resume_from_model_path)
      if ckpt and os.path.exists(ckpt.model_checkpoint_path):
        restore_path = ckpt.model_checkpoint_path
        restored_step = int(ckpt.model_checkpoint_path.split('-')[-1])
      else:
        logger.fatal('No restorable checkpoint model found.')
    if restore_path:
      restorer = tf.train.Saver(tf.all_variables())
      restorer.restore(sess, restore_path)
      logger.info('Restoring model from %s', restore_path)

  if mode == 'eval':
    restorer = tf.train.Saver(tf.all_variables())
    restorer.restore(sess, model_path)

  tf.get_default_graph().finalize()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  if mode == 'train':
    best_model_loss = 100.
    best_model_path = ""
    model_save_freq = ckpt_epoch_freq * num_batches_per_epoch
    logger.info('Save model per iters: {}'.format(model_save_freq))
    # train loop
    for i in range(num_iters_to_run):
      t_loss, niters, lr = sess.run([train_op, global_step, learning_rate])
      mean_loss_acc += t_loss
      if niters % 20 == 0:
        logger.info(print_msg(niters, lr, t_loss, mean_loss_acc / i))
      if niters % (model_save_freq) == 0:
        # completed epoch, save model snapshot
        _path = saver.save(sess, save_model_path, global_step=niters)
        if t_loss < best_model_loss:
          if os.path.exists(best_model_path):
            os.remove(best_model_path)
          best_model_loss = t_loss
          best_model_path = '{}-best-{:.2f}-{}'.format(
              save_model_path, t_loss, niters)
          shutil.copy(_path, best_model_path)
    # save the final model
    saver.save(sess, save_model_path, global_step=niters)

  elif mode == 'eval':
    logger.info('Evaluating model: %s', model_path)
    # eval loop
    sum_losses = 0.
    sum_weights = 0
    cap_gens = {}
    for i in xrange(num_iters_to_run):
      tgs, logi, los, im_id, eval_cross_entropy_loss, eval_weights = sess.run(
          [targets, logits, losses, image_id,
           target_cross_entropy_losses, target_cross_entropy_loss_weights])

      cap_gens[i] = {
          'true': tgs,
          'pred': np.argmax(logi, 1),
          'logi': logi,
          'eval_loss': eval_cross_entropy_loss,
          'mask': eval_weights,
          'id': im_id,
          }
      # each image is evaluated 5 times
      eval_weights /= 5.0
      sum_losses += np.sum(eval_cross_entropy_loss * eval_weights)
      sum_weights += np.sum(eval_weights)

      if i % 500 == 0:
        logger.info("Computed loss for %d/%d batches", i, num_iters_to_run)

    # write the most probable captions to file for computing other metrics
    save_predicted_caps_path = FLAGS.out_caption_path
    with open(save_predicted_caps_path, 'wb') as fp:
      import cPickle as pickle

      pickle.dump(cap_gens, fp)
    logger.info("Writing predicted caps to %s", save_predicted_caps_path)

    perplexity = math.exp(sum_losses / sum_weights)
    logger.info("Perplexity = %.2f", perplexity)

  coord.request_stop()
  coord.join(threads)
  sess.close()


if __name__ == '__main__':
  tf.app.run()

