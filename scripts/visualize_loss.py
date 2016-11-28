#/usr/bin/env python
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import re
import sys
import json
import argparse
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


MAX_SAMPLES = 50


def plot_training_curves(history, file_path, x_train=None, x_test=None,
                         done=True):
    fig, ax2 = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.set_tight_layout(None)
    x = np.arange(len(history['lr'])) + 1
    x_train = x if x_train is None else x_train
    x_test = x if x_test is None else x_test

    handles = []
    ax2.set_ylim([0., max(history['loss']) + 1.])
    if history.has_key('val_loss'):
        pvl, = ax2.plot(x_test, np.asarray(history['val_loss']),
                        'ro-', label='test loss')
        handles.append(pvl)
    if history.has_key('loss'):
        pl, = ax2.plot(x_train, np.asarray(history['loss']),
                       'mo-', label='loss')
        handles.append(pl)
    ax2.set_ylabel('Train/Test Loss', color='m')
    for tl in ax2.get_yticklabels():
        tl.set_color('m')

    ax1 = ax2.twinx()
    ax1.set_ylim([0., 1.])
    if history.has_key('val_loss'):
        pva, = ax1.plot(x_test, np.asarray(history['val_loss']),
                        'bo-', label='test lr')
        handles.append(pva)
    if history.has_key('lr'):
        pa, = ax1.plot(x_train, np.asarray(history['lr']),
                       'co-', label='lr')
        handles.append(pa)
    if done:
        ax1.set_xlabel('Epochs')
    else:
        ax1.set_xlabel('Iterations')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Learning Rate', color='c')
    for tl in ax1.get_yticklabels():
        tl.set_color('c')

    plt.legend(handles=handles)
    fig.savefig(file_path)


def main_complete_training_curve(args):
    fp = open(args.log_file, 'r')
    lines = fp.read()
    fp.close()
    if len(lines) == 0:
        sys.exit(0)
    txt = lines.replace('\n', ' ')
    pat = re.compile(r'\{.+\}')
    hist_str = pat.findall(txt)
    if len(hist_str) > 0:
        hist_str = hist_str[-1]
        hist_str = hist_str.replace("'", "\"")
        history = json.loads(hist_str)
        plot_training_curves(history, args.out_name)
        return


def main_progress_training_curve(args):
    fp = open(args.log_file, 'r')
    lines = fp.read()
    fp.close()
    if len(lines) == 0:
        sys.exit(0)
    txt = lines.replace('\n', ' ')
    ptrn_train = re.compile(
        r'niter: ([0-9]+) lr: ([0-9\.]+) batch_loss: ([0-9\.]+)')
    train_logs = ptrn_train.findall(txt)
    train_logs = [(int(i), float(lr), float(l))
                  for i, lr, l in train_logs]
    if len(train_logs) == 0:
        return
    train_sampling = int(ceil(len(train_logs) / MAX_SAMPLES))
    if train_sampling > 0:
        train_logs = train_logs[::train_sampling]
    history = {'lr': map(lambda k: k[1], train_logs),
               'loss': map(lambda k: k[2], train_logs)}
    x_train = map(lambda k: k[0], train_logs)

    if False:
      # no test logs
      ptrn_test = re.compile(r'epoch ([0-9]+), iter ([0-9]+), '
                             'test lr ([0-9\.]+), test loss ([0-9\.]+)')
      test_logs = ptrn_test.findall(txt)
      test_logs = [(int(e), int(i), float(a), float(l))
                   for e, i, a, l in test_logs]
      test_sampling = int(ceil(len(test_logs) / MAX_SAMPLES))
      if test_sampling > 0:
          test_logs = test_logs[::test_sampling]
      x_test = None
      if len(test_logs) != 0:
          history['val_loss'] = map(lambda k: k[-2], test_logs)
          history['val_loss'] = map(lambda k: k[-1], test_logs)
          x_test = map(lambda k: k[1], test_logs)

    plot_training_curves(history, args.out_name, x_train, done=False)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('out_name')
    parser.add_argument('-done', dest='done', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arguments()
    if args.done:
        main_complete_training_curve(args)
    else:
        main_progress_training_curve(args)
