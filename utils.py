import logging
from addict import Dict

logging.basicConfig(
    format="[\x1b[1;36m%(levelname)s:%(filename)s"
           "|L%(lineno)s\x1b[0m] %(message)s")

CONFIG = Dict()

CONFIG.logLevel = logging.INFO
CONFIG.GPUs = 3

#
# process images and store the CNN features
#
CONFIG.DatasetLoader.logLevel = CONFIG.logLevel
CONFIG.DatasetLoader.par_jobs = -2

CONFIG.ResNetFeatureExtractor.logLevel = CONFIG.logLevel
CONFIG.ResNetFeatureExtractor.input_layer = 'images:0'
CONFIG.ResNetFeatureExtractor.output_layer = 'avg_pool:0'
CONFIG.ResNetFeatureExtractor.batch_size = 64
CONFIG.ResNetFeatureExtractor.feat_size = 2048

CONFIG.VGGFeatureExtractor.logLevel = CONFIG.logLevel
# CONFIG.VGGFeatureExtractor.input_layer = 'images:0'
CONFIG.VGGFeatureExtractor.output_layer = 'import/fc7/Reshape:0'
CONFIG.VGGFeatureExtractor.batch_size = 64
CONFIG.VGGFeatureExtractor.feat_size = 4096

#
# process captions and train a lstm to build a language model
#
CONFIG.CapData.logLevel = logging.DEBUG

#
# model related configs
#
CONFIG.Model.logLevel = logging.DEBUG

