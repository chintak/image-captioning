import logging
from addict import Dict


def _setup_logger(flag=0, name=None, fname=None, fmt=None, fmode='w'):
  """Setup logger

  flag:
    1 - use only console
    2 - use only file
    3 - use console and file
  """
  if fmt is None:
    fmt = "\x1b[1;36m%(levelname)s\x1b[0m %(message)s"
  _logger = logging.getLogger(name)
  if flag not in [1, 2, 3]:
    flag = 1
    _logger.info('Invalid logger config; using console only')
  _formatter = logging.Formatter(fmt)
  if flag & 1:
    _streamer = logging.StreamHandler()
    _streamer.setFormatter(_formatter)
    _logger.addHandler(_streamer)
  if flag & 2:
    if fname is None:
      fname = 'run.log'
    _formatter_with_time = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    _filer = logging.FileHandler(fname, mode=fmode)
    _filer.setFormatter(_formatter_with_time)
    _logger.addHandler(_filer)
  return _logger


CONFIG = Dict()

CONFIG.log.level = logging.INFO
CONFIG.log.getLogger = _setup_logger
CONFIG.GPUs = 3

#
# process images and store the CNN features
#
CONFIG.DatasetLoader.log = CONFIG.log
# CONFIG.DatasetLoader.log.level = logging.DEBUG
CONFIG.DatasetLoader.par_jobs = -2

CONFIG.ResNetFeatureExtractor.log = CONFIG.log
CONFIG.ResNetFeatureExtractor.input_layer = 'images:0'
CONFIG.ResNetFeatureExtractor.output_layer = 'avg_pool:0'
CONFIG.ResNetFeatureExtractor.batch_size = 64
CONFIG.ResNetFeatureExtractor.feat_size = 2048

CONFIG.VGGFeatureExtractor.log = CONFIG.log
# CONFIG.VGGFeatureExtractor.input_layer = 'images:0'
CONFIG.VGGFeatureExtractor.output_layer = 'import/fc7/Reshape:0'
CONFIG.VGGFeatureExtractor.batch_size = 64
CONFIG.VGGFeatureExtractor.feat_size = 4096

#
# process captions and train a lstm to build a language model
#
CONFIG.CapData.log = CONFIG.log
CONFIG.CapData.log.level = logging.DEBUG
CONFIG.CapData.max_length = 25
CONFIG.CapData.min_freq = 5

#
# model related configs
#
CONFIG.Trainer.log = CONFIG.log
CONFIG.Trainer.log.level = logging.DEBUG

CONFIG.Decoder.log = CONFIG.log

