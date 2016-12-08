#
# Inspired from https://www.quora.com/How-do-I-visualise-word2vec-word-vectors/answer/Vered-Shwartz?srid=pp1F
#
from os.path import join, splitext
import sys
import codecs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import nltk

colors = {
  'NN': 'blue',
  'JJ': 'red',
  'VB': 'cyan',
  'RB': 'green',
}

def main():
  embeddings_file = sys.argv[1]
  save_path = '{}.eps'.format(splitext(embeddings_file)[0])
  wv, vocabulary = load_embeddings(embeddings_file)

  tsne = TSNE(n_components=2, random_state=0)
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(wv[:1000,:])

  fig = plt.figure(figsize=(100.0, 100.0))
  fig.add_subplot(111).scatter(Y[:, 0], Y[:, 1])
  for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
    _, pt = nltk.pos_tag([label])[0]
    for k in colors.keys():
      if k in pt:
        pt = k
        break
    fig.add_subplot(111).annotate(label, xy=(x, y), xytext=(0, 0),
                                  textcoords='offset points',
                                  color=colors.get(pt, 'black'))
  fig.savefig(save_path, format='eps', dpi=1000)
  print "Saved at %s" % save_path


def load_embeddings(file_name):
  with codecs.open(file_name, 'r', 'utf-8') as f_in:
    vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
  wv = np.loadtxt(wv)
  return wv, vocabulary


if __name__ == '__main__':
  main()
