from __future__ import print_function

class CapConfig(object):

    def __init__(self, use_features_or_model=True, image_features_path=None,
                 image_model_path=None):
        # common settings
        self.batch_size = 32

        # initializer for weight matrix for image and word embedding layer
        self.initializer_scale = 0.08

        # embedding dim for image and word
        self.embedding_dim = 512

        # use offline extracted image features or extract features now
        self.use_features_or_model = use_features_or_model
        self.image_model_path = image_model_path  # used mainly for inference
        self.image_features_path = image_features_path
        assert ((self.use_features_or_model and self.image_features_path) or
                (not self.use_features_or_model and self.image_model_path)), (
                "image_features_path should be provided with"
                " use_features_or_model=True")

        # input image size expected by VGG16
        self.image_height = 224
        self.image_width = 244

        # caption related configs

        # max number of words in the vocabulary + 1 for <unk>
        # this can be set to a value larger than the actual number of words
        self.vocab_size = 10000

        self.num_lstm_cells = 512

        self.lstm_dropout_prob = 0.7


class TrainConfig(object):

    def __init__(self):
        # the optimizer for training the model
        self.optimizer = 'Adam'

        self.init_learning_rate = 2.0
        self.decay_learning_rate = 0.5
        self.num_epochs_per_decay = 8

        # useful while training LSTM
        self.clip_gradients = 5.0

        self.max_chkpts_to_keep = 3

