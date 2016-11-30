"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import
from __future__ import division

import os
from os.path import join, exists
import sys
import math
import shutil
import tensorflow as tf
import numpy as np
from addict import Dict
from pprint import pformat, pprint

# from external.inputs import batch_with_dynamic_pad
from utils import CONFIG

config = CONFIG.Trainer
logger = None

###############################################################################
###############################################################################
###############################################################################

class ImCapModel(object):
  """docstring for ImCapModel"""
  def __init__(self, configures, word2int, log_fname=None):
    global logger

    flags = Dict(configures)
    logger = config.log.getLogger(
        flag=3 if log_fname is not None else 1, fname=log_fname)
    logger.setLevel(config.log.level)
    # setup default configurations
    assert flags.mode in ['train', 'eval', 'inference']
    assert flags.num_samples and flags.mode != 'inference'
    flags.batch_size = (flags.get('batch_size', None)
                        if flags['mode'] == 'train' else 1)
    flags.ckpt_epoch_freq = flags.get('ckpt_epoch_freq', 5)
    # flags.num_reader_threads = flags.get('num_reader_threads', 4)

    initializer_scale = 0.08
    flags.params_initializer = tf.random_uniform_initializer(
        minval=-initializer_scale, maxval=initializer_scale)

    # placeholder for feeding image features
    self.images = None
    self.image_embeddings = None
    flags.img_input_feed = flags.get('img_input_feed', 'image_feed')
    self.image_feed = flags.img_input_feed
    flags.img_feature_length = 4096
    flags.img_embedding_size = 512

    # placeholder for feeding captions
    self.input_seqs = None
    self.seq_embedding_map = None
    self.seq_embeddings = None
    self.word2int = word2int
    flags.cap_input_feed = flags.get('cap_input_feed', 'input_feed')
    self.input_feed = flags.cap_input_feed
    flags.cap_ntime_steps = flags.get('time_steps', 25)
    flags.cap_vocab_size = flags.get('vocab_size', 9000)
    flags.cap_embedding_size = flags.get('embedding_size', 512)
    flags.num_lstm_units = flags.get('lstm_cells', 512)
    flags.lstm_dropout_prob = 0.7 if flags.get('dropout', True) else 0.0

    # special tokens
    self._pad = word2int['<PAD>']

    # setup the placeholders and other variables for use later
    self.flags = flags
    logger.info('Model configuration: %s', pformat(self.flags))

    # train, eval or inference
    self.mode = self.flags.mode

    # setup the LSTM cell for decoding the captions
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.flags.num_lstm_units)
    if self.mode == 'train':
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, input_keep_prob=self.flags.lstm_dropout_prob,
          output_keep_prob=self.flags.lstm_dropout_prob)

    self.cell = cell

  def _build_inputs(self):
    self.images = tf.placeholder(
        dtype=tf.float32,
        shape=[self.flags.batch_size, self.flags.img_feature_length],
        name=self.flags.img_input_feed)

    self.input_seqs = tf.placeholder(
        dtype=tf.int64,
        shape=[self.flags.batch_size, self.flags.cap_ntime_steps + 1],
        name=self.flags.cap_input_feed)

  def _build_image_embed_layer(self):
    # image features are stored in image_embeddings with shape [batch_size,
    # embedding_size]
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          self.images,
          num_output_units=self.flags.img_embedding_size,
          activation_fn=None,
          weight_init=self.flags.params_initializer,
          bias_init=None,
          name=scope)

    self.image_embeddings = image_embeddings
    logger.debug("Setting up the image embedding layer")

  def _build_word_embed_layer(self):
    # caption features are stored in seq_embeddings with shape [batch_size,
    # seq_length, embedding_size]
    with tf.variable_scope("seq_embedding"), tf.device('/cpu:0'):
      seq_embedding_map = tf.get_variable(
          name="map",
          shape=[self.flags.cap_vocab_size, self.flags.cap_embedding_size],
          initializer=self.flags.params_initializer)
      seq_embeddings = tf.nn.embedding_lookup(seq_embedding_map,
                                              self.input_seqs)

    self.seq_embedding_map = seq_embedding_map
    self.seq_embeddings = seq_embeddings
    logger.debug("Setting up the word embedding layer")

  def _initialize_lstm(self, image_embed_feats):
    with tf.variable_scope(
          'lstm',initializer=self.flags.params_initializer) as lstm_scope:
      # initialize the lstm with image embeddings
      zero_state = self.cell.zero_state(
          batch_size=image_embed_feats.get_shape()[0], dtype=tf.float32)
      m, initial_state = self.cell(image_embed_feats, zero_state)
    return initial_state

  def build_model(self):
    self._build_inputs()
    self._build_image_embed_layer()
    self._build_word_embed_layer()

    word_captions = self.input_seqs
    enc_captions = self.seq_embeddings
    input_sequence = enc_captions[:, :self.flags.cap_ntime_steps, :]
    sequence_mask = tf.to_float(tf.not_equal(word_captions, self._pad))

    state_list = self._initialize_lstm(self.image_embeddings)
    indices = tf.to_int64(tf.expand_dims(
        tf.range(0, self.flags.batch_size, 1), 1))
    one_hot_map_size = [self.flags.batch_size, self.flags.cap_vocab_size]

    loss = 0.0

    for t in range(self.flags.cap_ntime_steps):
      labels = tf.expand_dims(word_captions[:, t], 1)
      idx_to_labs = tf.concat(1, [indices, labels])
      target_sequence = tf.sparse_to_dense(
          idx_to_labs, tf.to_int64(tf.pack(one_hot_map_size)), 1.0, 0.0)

      with tf.variable_scope(
          'lstm', initializer=self.flags.params_initializer, reuse=True):
        # Run a single LSTM step.
        m, state_list = self.cell(input_sequence[:, t, :], state=state_list)

      with tf.variable_scope('logits', reuse=(t!=0)) as logits_scope:
        w_o = tf.get_variable('w', shape=[self.flags.num_lstm_units,
                                          self.flags.cap_vocab_size])
        b_o = tf.get_variable('b', shape=[self.flags.cap_vocab_size])

        logits = tf.matmul(m, w_o) + b_o

        softmax = tf.nn.softmax_cross_entropy_with_logits(
            logits, target_sequence)
        loss += tf.reduce_sum(softmax * sequence_mask[:, t])

    return loss / tf.to_float(self.flags.batch_size)

  def build_generator(self):
    self._build_inputs()
    self._build_image_embed_layer()
    self._build_word_embed_layer()
    initial_state = self._initialize_lstm(self.image_embeddings)

    with tf.variable_scope(
          'lstm',initializer=self.flags.params_initializer) as lstm_scope:
      lstm_scope.reuse_variables()

      # Placeholder for feeding a batch of concatenated states.
      state_feed = tf.placeholder(
          dtype=tf.float32,
          shape=[None, self.cell.state_size],
          name="state_feed")
      state_tuple = tf.split(1, 2, state_feed)


###############################################################################
###############################################################################
###############################################################################

class Solver(object):
  """docstring for Solver"""
  def __init__(self, flags):
    flags.num_epochs = flags['num_epochs']
    flags.clip_gradients = 5.0
    flags.opt_name = flags.get('optimizer', 'SGD')
    if flags.opt_name.lower() == 'sgd':
      flags.lr_initial = flags.get('learning_rate', 2.0)
    elif flags.opt_name.lower() == 'adam':
      flags.lr_initial = flags.get('learning_rate', 0.001)
    else:
      raise AttributeError('learning_rate expects SGD or Adam')
    flags.lr_nepoch_per_decay = flags.num_epochs // 5
    # use the following params for piecewise constant lr decay
    flags.lr_ndecay_steps = 5
    # self.flags.lr_decay_values: [2.0 1.0 0.5 0.1]
    flags.lr_decay_vals = str_to_real_list(flags.lr_decay_values, ',')

    assert exists(flags.model_path)
    flags.model_path = (tf.train.latest_checkpoint(flags.model_path)
        if os.path.isdir(flags.model_path) else flags.model_path)
    assert flags.out_caption_path
    assert flags.save_model_dir
    if not exists(flags.save_model_dir):
      os.makedirs(flags.save_model_dir)
    flags.save_model_path = join(flags.save_model_dir, 'model')


###############################################################################
###############################################################################
###############################################################################

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
  if not sval:
    return None
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

###############################################################################
###############################################################################
###############################################################################

def main(_):
  global logger

  mode = self.mode

  logger = config.log.getLogger(
      flag=3 if mode == 'train' else 1, fname=log_fname)
  logger.setLevel(config.log.level)
  # print the experiment flags for logging purpose
  logger.info("python %s", stringify(sys.argv, ' '))

  # Config variables
  c = 0
  for record in tf.python_io.tf_record_iterator(self.flags.data_path):
     c += 1
  num_samples = c
  num_epochs = self.flags.num_epochs if self.mode == 'train' else 1
  batch_size = 128 if mode == 'train' else 1
  num_reader_threads = 4
  ckpt_epoch_freq = 5

  initializer_scale = 0.08
  initializer = tf.random_uniform_initializer(
      minval=-initializer_scale,
      maxval=initializer_scale)

  img_feature_length = 4096

  vocab_size = 10000
  embedding_size = 512
  num_lstm_units = 512
  lstm_dropout_prob = 0.7

  vals_queue_name = 'value_queue'
  image_feature = 'image_feature'
  caption_feature = 'caption_feature'

  train_clip_gradients = 5.0
  self.flags.optim.name = self.flags.optimizer
  if self.flags.optim.name == 'SGD':
    initial_learning_rate = 2.0
  else:
    initial_learning_rate = .001
  lr_decay_method = self.flags.lr_decay_method
  # use the following params for exponential lr decay
  learning_rate_decay_rate = 0.5
  num_epochs_per_decay = min(num_epochs // 5, 10)
  # use the following params for piecewise constant lr decay
  num_decay_steps = 5
  # self.flags.lr_decay_values: [2.0 1.0 0.5 0.1]
  lr_decay_values = str_to_real_list(self.flags.lr_decay_values, ',')

  # setup the inputs
  if mode == "inference":
    images = tf.placeholder(dtype=tf.float32,
                            shape=[img_feature_length],
                            name="image_feed")
    input_feed = tf.placeholder(dtype=tf.int64,
                                shape=[None],  # batch_size
                                name="input_feed")

    # Process image and insert batch dimensions.
    input_seqs = tf.expand_dims(input_feed, 1)

    # No target sequences or input mask in inference mode.
    target_seqs = None
    input_mask = None
  else:
    filename_queue = tf.train.string_input_producer([self.flags.data_path])

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

      image, input_cap = parse_seq_example(serialized_sample,
                                           image_feature, caption_feature)
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
        initializer=self.flags.params_initializer)
    seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

  # image features are stored in image_embeddings with shape [batch_size,
  # embedding_size]
  # caption features are stored in seq_embeddings with shape [batch_size,
  # seq_length, embedding_size]

  # create lstm cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_lstm_units)
  if mode == 'train':
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=lstm_dropout_prob,
        output_keep_prob=lstm_dropout_prob)

  with tf.variable_scope('lstm', initializer=self.flags.params_initializer) as lstm_scope:
    # initialize the lstm with image embeddings
    zero_state = cell.zero_state(
        batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
    _, initial_state = cell(image_embeddings, zero_state)

    lstm_scope.reuse_variables()

    if mode == 'inference':
      tf.concat(1, initial_state, name="initial_state")

      # Placeholder for feeding a batch of concatenated states.
      state_feed = tf.placeholder(dtype=tf.float32,
                                  shape=[None, sum(cell.state_size)],
                                  name="state_feed")
      state_tuple = tf.split(1, 2, state_feed)

      # Run a single LSTM step.
      lstm_outputs, state_tuple = cell(
          inputs=tf.squeeze(self.seq_embeddings, squeeze_dims=[1]),
          state=state_tuple)

      # Concatentate the resulting state.
      tf.concat(1, state_tuple, name="state")
    else:
      seq_lengths = tf.reduce_sum(input_mask, 1)
      lstm_outputs, _ = tf.nn.dynamic_rnn(
          cell=cell, inputs=seq_embeddings,
          sequence_length=seq_lengths, initial_state=initial_state,
          dtype=tf.float32, scope=lstm_scope)

  # stack batches vertically
  lstm_outputs = tf.reshape(lstm_outputnames, [-1, cell.output_size])

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
  num_batches_per_epoch = num_samples // batch_size + 1
  num_iters_to_run = num_epochs * num_batches_per_epoch
  num_epochs_done = 0
  mean_loss_acc = 0.

  # set up learing rate decay and learning rate decay
  learning_rate_decay_fn = None
  if self.flags.optim.name.lower() == 'sgd' and lr_decay_method == 'piecewise':
    num_steps = len(lr_decay_values)
    if self.flags.lr_decay_boundaries:
      lr_vs = str_to_int_list(self.flags.lr_decay_boundaries, ',')
      lr_decay_boundaries = [num_batches_per_epoch * i for i in lr_vs]
    else:
      batches_per_decay = (num_batches_per_epoch * num_epochs) // num_steps
      lr_decay_boundaries = [batches_per_decay * i
                             for i in range(1, num_steps)]
    learning_rate = piecewise_constant(
          global_step,
          lr_decay_boundaries,
          lr_decay_values)
    logger.info('Learning rate with piecewise constant decay %s - %s',
                stringify(lr_decay_boundaries), stringify(lr_decay_values))
  elif (self.flags.optim.name.lower() == 'sgd' and lr_decay_method == 'exp' and
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
      optimizer=self.flags.optim.name,
      clip_gradients=train_clip_gradients,
      learning_rate_decay_fn=learning_rate_decay_fn)

  # setup saver
  saver = tf.train.Saver(max_to_keep=0)

  # setup Session and begin training
  sess = tf.Session()
  coord = tf.train.Coordinator()

  init = tf.initialize_all_variables()
  sess.run(init)

  logger.info('TFRecord file: {}'.format(self.flags.data_path))
  if mode == 'train':
    logger.info('Mode save path: {}'.format(self.flags.save_model_dir))
    logger.info('Initial learning rate: {}'.format(initial_learning_rate))
  logger.info('Num of samples: {}'.format(num_samples))
  logger.info('Num of iters: {}'.format(num_iters_to_run))

  # restore previously saved model to resume training
  if self.flags.resume_from_model_path:
    restore_path = None
    if os.path.isfile(self.flags.resume_from_model_path):
      restore_path = self.flags.resume_from_model_path
    else:
      ckpt = tf.train.get_checkpoint_state(self.flags.resume_from_model_path)
      if ckpt and exists(ckpt.model_checkpoint_path):
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
          if exists(best_model_path):
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
    save_predicted_caps_path = self.flags.out_caption_path
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

