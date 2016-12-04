import os
import sys
sys.path.append('../')
sys.path.append('external/')
sys.path.append('../external/')
import cPickle as pickle
import argparse
from pprint import pformat

from score import get_score
from utils import CONFIG

config = CONFIG.Decoder
logger = config.log.getLogger(flag=3, fname='report.txt', fmode='a')

def decode_samples_to_captions(samples, id_to_word):
  captions = []
  for sample in samples:
    cap = []
    for idx in sample:
      w = id_to_word.get(idx, None)
      if w is None or w == '<ST>':
        continue
      elif w == '<ET>':
        break
      cap.append(w)
    captions.append(cap)
  return [' '.join(cap) for cap in captions]

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
for i, val in enumerate(caps):
  true_cap = decode_samples_to_captions([val['true']], ivocab)[0]
  pred_cap = decode_samples_to_captions([val['pred']], ivocab)[0]

  if hyp.get(val['id'], None) and len(hyp[val['id']]) < 5:
    hyp[val['id']].append(pred_cap)
    ref[val['id']].append(true_cap)
  else:
    hyp[val['id']] = [pred_cap]
    ref[val['id']] = [true_cap]
  decoded_caps[val['id']] = {
      'true': ref[val['id']],
      'pred': hyp[val['id']]
      }

for k, v in hyp.iteritems():
  hyp[k] = [v[0]]

print ref.items()[:10]
print ""
print hyp.items()[:10]
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

