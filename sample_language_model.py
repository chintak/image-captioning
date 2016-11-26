"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import

import os
import logging
import argparse
import tensorflow as tf

from reader import flickr8k_raw_data
from configuration import CapConfig
from external.inputs import batch_with_dynamic_pad
from utils import CONFIG

config = CONFIG.Trainer
logger = None


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_tfrecord_path", "",
                       "Path to training data in TFRecord format.")
tf.flags.DEFINE_string("save_model_path", "",
                       "Path to dir to save model checkpoint.")
tf.flags.DEFINE_integer("num_epochs", 10,
                        "Num of epochs for training the model.")


def parse_seq_example(serialized, image_feature, caption_feature):
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={
      image_feature: tf.FixedLenFeature([4096], dtype=tf.float32)
    },
    sequence_features={
      caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64)
  })

  return context[image_feature], sequence[caption_feature]


def print_msg(*args):
  return "niter: {} lr: {} batch_loss: {} avg_batch_loss: {}".format(*args)


def main(_):
  global logger
  # args = arguments()
  assert FLAGS.train_tfrecord_path
  assert FLAGS.save_model_path

  logger = config.log.getLogger(
      flag=3, fname='{}.log'.format(os.path.join(FLAGS.save_model_path, 'run')))
  logger.setLevel(config.log.level)

  # Config variables
  mode = 'train'
  num_train_samples = 50000
  num_epochs = FLAGS.num_epochs
  batch_size = 128
  num_reader_threads = 4

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
  optimizer_name = 'SGD'
  if optimizer_name == 'SGD':
    initial_learning_rate = 2.0
  else:
    initial_learning_rate = .05
  learning_rate_decay_rate = 0.5
  num_epochs_per_decay = num_epochs // 5

  if not os.path.exists(FLAGS.save_model_path):
    os.mkdir(FLAGS.save_model_path)
  save_model_path = os.path.join(FLAGS.save_model_path, 'model')

  # setup the inputs
  filename_queue = tf.train.string_input_producer([FLAGS.train_tfrecord_path])

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
  # TODO: add an embedding layer for cnn features

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
  learning_rate = tf.constant(initial_learning_rate)
  learning_rate_decay_fn = None
  if learning_rate_decay_rate > 0:
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    learning_rate_decay_fn = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps,
        learning_rate_decay_rate,
        staircase=True)

  # now setup the optimizer for training the model
  train_op = tf.contrib.layers.optimize_loss(
      loss=batch_loss,
      global_step=global_step,
      learning_rate=learning_rate,
      optimizer=optimizer_name,
      clip_gradients=train_clip_gradients)

  # setup saver
  saver = tf.train.Saver(max_to_keep=5)

  # setup Session and begin training
  sess = tf.Session()
  coord = tf.train.Coordinator()

  init = tf.initialize_all_variables()
  sess.run(init)
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  logger.info('TFRecord file: {}'.format(FLAGS.train_tfrecord_path))
  logger.info('Mode save path: {}'.format(FLAGS.save_model_path))
  logger.info('Initial learning rate: {}'.format(initial_learning_rate))
  logger.info('Num of samples: {}'.format(num_train_samples))
  logger.info('Num of iters: {}'.format(num_iters_to_run))
  logger.info('Save model per iters: {}'.format(num_batches_per_epoch))

  # train loop
  for i in range(num_iters_to_run):
    t_loss, niters, lr = sess.run([train_op, global_step, learning_rate])
    mean_loss_acc += t_loss
    if niters % 20 == 0:
      logger.info(print_msg(niters, lr, t_loss, mean_loss_acc / niters))
    if niters % num_batches_per_epoch == 0:
      # completed epoch, save model snapshot
      saver.save(sess, save_model_path, global_step=niters)

  coord.request_stop()
  coord.join(threads)
  sess.close()


def arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('train_tfrecord_path', help='Path to train TFRecord.')
  parser.add_argument('--save_model_path', help='Path to save model ckpt.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  tf.app.run()

