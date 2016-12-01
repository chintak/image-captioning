"""Builda sample language model with LSTM and text captions."""
from __future__ import absolute_import
from __future__ import division

import os
from os.path import join, exists
import tensorflow as tf
import numpy as np
from addict import Dict
from pprint import pformat

from utils import CONFIG
config = CONFIG.Trainer
logger = None


class ImCapModel(object):
  """docstring for ImCapModel"""
  def __init__(self, configures, word2int, log_fname=None):
    global logger

    flags = Dict(configures)
    logger = config.log.getLogger(
        flag=2 if flags.log_fname is not None else 1,
        fname=flags.log_fname, fmode='a')
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
    self._start = word2int['<ST>']
    self._end = word2int['<ET>']

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

  def _build_word_embed_layer(self, input_seq, reuse=False):
    # caption features are stored in seq_embeddings with shape [batch_size,
    # seq_length, embedding_size]
    with tf.variable_scope("seq_embedding", reuse=reuse), tf.device('/cpu:0'):
      seq_embedding_map = tf.get_variable(
          name="map",
          shape=[self.flags.cap_vocab_size, self.flags.cap_embedding_size],
          initializer=self.flags.params_initializer)
      seq_embeddings = tf.nn.embedding_lookup(seq_embedding_map, input_seq)
    return seq_embeddings

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
    self.seq_embeddings = self._build_word_embed_layer(self.input_seqs)

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
    # assume inputs and image embedding already setup
    state_list = self._initialize_lstm(self.image_embeddings)

    generated_words = []
    start_word = tf.expand_dims(tf.convert_to_tensor([self._start]), 1)
    sample_word = tf.tile(start_word, [self.flags.batch_size, 1])

    for t in range(self.flags.cap_ntime_steps):
      seq_embeddings = self._build_word_embed_layer(sample_word)
      with tf.variable_scope(
          'lstm', initializer=self.flags.params_initializer, reuse=True):
        # Run a single LSTM step.
        m, state_list = self.cell(tf.squeeze(seq_embeddings), state=state_list)

      with tf.variable_scope('logits', reuse=(t!=0)) as logits_scope:
        w_o = tf.get_variable('w', shape=[self.flags.num_lstm_units,
                                          self.flags.cap_vocab_size])
        b_o = tf.get_variable('b', shape=[self.flags.cap_vocab_size])

        logits = tf.matmul(m, w_o) + b_o
        sample_word = tf.argmax(logits, 1)
        generated_words.append(sample_word)

    generated_captions = tf.transpose(
        tf.pack(generated_words), (1, 0), name=self.flags.cap_generated)
    return generated_captions
