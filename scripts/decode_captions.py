import os
import sys
sys.path.append('../')
sys.path.append('external/')
sys.path.append('../external/')
import cPickle as pickle
import argparse
from pprint import pformat

from score import get_score


parser = argparse.ArgumentParser()
parser.add_argument('encoded_caps_path')
parser.add_argument('output_path')
parser.add_argument('vocab_path')
args = parser.parse_args()

assert os.path.exists(args.encoded_caps_path)
assert os.path.exists(args.vocab_path)
path = args.encoded_caps_path

with open(path, 'rb') as fp:
  caps = pickle.load(fp)

with open(args.vocab_path, 'rb') as fp:
  vocab = pickle.load(fp)

ivocab = dict([(v, k) for k, v in vocab.iteritems()])

hyp = {}
ref = {}
decoded_caps = {}
for i, val in enumerate(caps.values()):
  true_cap = ' '.join([ivocab[l] for l in val['true']])
  pred_cap = ' '.join([ivocab[l] for l in val['pred']])

  if hyp.get(val['id'], None) is not None:
    hyp[val['id']].append(pred_cap)
    ref[val['id']].append(true_cap)
  else:
    hyp[val['id']] = [pred_cap]
    ref[val['id']] = [true_cap]
  decoded_caps[val['id']] = {
      'true': ref[val['id']],
      'pred': hyp[va['id']]
      }

for k, v in hyp.iteritems():
  hyp[k] = [v[0]]
final_scores = get_score(ref, hyp)

print 'Bleu_1:\t',final_scores['Bleu_1']
print 'Bleu_2:\t',final_scores['Bleu_2']
print 'Bleu_3:\t',final_scores['Bleu_3']
print 'Bleu_4:\t',final_scores['Bleu_4']
print 'METEOR:\t',final_scores['METEOR']
print 'ROUGE_L:',final_scores['ROUGE_L']
print 'CIDEr:\t',final_scores['CIDEr']

out_path = args.output_path
with open(out_path, 'w') as fp:
  fp.write(pformat(decoded_caps))
  fp.write('\n')

print "Decoded captions written to", out_path

