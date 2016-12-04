import json
import os
from os.path import join
import scipy.io
import numpy as np
from utils import CONFIG

config = CONFIG.CapData
MAX_LENGTH = config.max_length

__all__ = ['load_dataset']


def _build_vocab(data):
  words = [w for ex in data for sen in ex['sentences'] for w in sen['tokens']]
  words = sorted(set(words) - set(['<PAD>', '<ST>', '<ET>']))
  vocab = {'<PAD>': 0, '<ST>': 1, '<ET>': 2}
  idx = 3
  for word in words:
    vocab[word] = idx
    idx += 1
  print "Total number of words in the vocabulary =", idx
  return vocab

def _prepare_captions(data, word_to_id):
  X_toks = []
  img_to_feat_id = {}
  cap_id_to_img = []

  num_of_samples = 0
  for ex in data:
    for sen in ex['sentences']:
      if len(sen['tokens']) <= MAX_LENGTH:
        num_of_samples += 1
  print "%d sentences found" % num_of_samples

  rng = np.random.RandomState(seed=1234)
  idxs = rng.permutation(num_of_samples)

  for ex in data:
    img_name = ex['filename']
    img_id = ex['imgid']
    for sen in ex['sentences']:
      toks = np.ones((1, MAX_LENGTH + 2), dtype=np.int64) * word_to_id['<PAD>']
      toks[0, 0] = word_to_id['<ST>']
      idx = 1
      if len(sen['tokens']) > MAX_LENGTH:
        continue
      for w in sen['tokens']:
        toks[0, idx] = word_to_id[w]
        idx += 1
      toks[0, idx] = word_to_id['<ET>']
      X_toks.append(toks)
      cap_id_to_img.append(img_name)
      img_to_feat_id[img_name] = img_id
  X_toks = np.concatenate(X_toks)
  X_toks = X_toks[idxs, :]
  cap_id_to_img = np.asarray(cap_id_to_img)
  cap_id_to_img = cap_id_to_img[idxs]

  return (X_toks, cap_id_to_img, img_to_feat_id)

def load_dataset(folder_path, word_to_id=None, split='train'):
  """This wrapper is for returning the VGG features and captions which can
  be used for training."""
  assert word_to_id or split == 'train'

  feats_mat_path = join(folder_path, 'vgg_feats.mat')
  dataset_path = join(folder_path, 'dataset.json')

  feats = scipy.io.loadmat(feats_mat_path)['feats']
  cnn_features = feats.T

  with open(dataset_path, 'r') as fp:
    data = json.load(fp)['images']

  if not word_to_id:
    word_to_id = _build_vocab(data)

  data = [ex for ex in data if ex['split'] == split]
  (capts, cap_id_to_img, img_to_feat_id) = _prepare_captions(data, word_to_id)

  return (capts, cap_id_to_img, cnn_features, img_to_feat_id, word_to_id)
