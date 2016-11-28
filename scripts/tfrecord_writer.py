import os
import argparse
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')

from reader import flickr8k_raw_data

def make_example(image_feature, caption_feature, id):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(caption_feature)
    for f in image_feature:
        ex.context.feature["image_feature"].float_list.value.append(float(f))
    ex.context.feature["id"].bytes_list.value.append(id)
    fl_tokens = ex.feature_lists.feature_list["caption_feature"]
    for token in caption_feature:
        fl_tokens.feature.add().int64_list.value.append(token)
    return ex


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('cnn_feats_path', help='a numpy.mmap expected')
    parser.add_argument(
        'caption_tokens_dir',
        help='Directory containing train, test and dev captions.')
    args = parser.parse_args()
    return args


def _strip_name(paths):
    return [(os.path.basename(p), i) for i, p in enumerate(paths)]


def main():
    args = arguments()
    # read the mmap file containing CNN features
    feats_fname = os.path.splitext(os.path.basename(args.cnn_feats_path))[0]
    img_name_list_path = os.path.join(
        os.path.dirname(args.cnn_feats_path),
        '{}_list.txt'.format(
            '_'.join(feats_fname.split('_')[:-3])))
    feats_shape = tuple([int(i) for i in feats_fname.split('_')[-1].split('X')])
    feats_mmap = np.memmap(args.cnn_feats_path, mode='r',  # read-only
                           shape=feats_shape, dtype=np.float32)
    img_to_idx = {}
    with open(img_name_list_path, 'r') as fp:
        img_to_idx = dict(_strip_name(fp.read().split('\n')))

    # load all the captions
    train_caps, test_caps, dev_caps, vocab = flickr8k_raw_data(
        args.caption_tokens_dir)
    rand_idx = np.arange(0, len(train_caps['names']))
    rng = np.random.RandomState(seed=1234)
    rng.shuffle(rand_idx)

    # dump the captions generated for debugging purpose
    with open(os.path.join(args.caption_tokens_dir, 'dump.txt'), 'w') as fp:
      from pprint import pformat

      fp.write("\n###### vocab######\n")
      fp.write(pformat(vocab))
      fp.write("\n###### train ######\n")
      rand_train_caps = {
        'names': [train_caps['names'][i] for i in rand_idx],
        'word_to_ids': [train_caps['word_to_ids'][i] for i in rand_idx],
      }
      fp.write(pformat([(n, w) for n, w in zip(
        rand_train_caps['names'], rand_train_caps['word_to_ids'])]))
      fp.write("\n###### test ######\n")
      fp.write(pformat([(n, w) for n, w in zip(
        test_caps['names'], test_caps['word_to_ids'])]))
      fp.write("\n###### dev ######\n")
      fp.write(pformat([(n, w) for n, w in zip(
        dev_caps['names'], dev_caps['word_to_ids'])]))

    # process train imgs and write to a record file
    train_tfrecord_name = os.path.join(
        args.caption_tokens_dir, '{}.train.tfrecord'.format(
            '_'.join(feats_fname.split('_')[:-3])))
    train_writer = tf.python_io.TFRecordWriter(train_tfrecord_name)
    # for i, (img_name, cap_ids) in enumerate(
    #         zip(train_caps['names'], train_caps['word_to_ids'])):
    for i, (idx) in enumerate(rand_idx):
        img_name = train_caps['names'][idx].split('#')[0]
        cap_ids = train_caps['word_to_ids'][idx]
        img_feat = feats_mmap[img_to_idx[img_name], :]
        train_writer.write(
            make_example(img_feat, cap_ids, img_name).SerializeToString())
        if i % 100 == 0:
            print "train records written {}/{}".format(
                    i, len(train_caps['names']))
    train_writer.close()

    # process test imgs and write to a record file
    test_tfrecord_name = os.path.join(
        args.caption_tokens_dir, '{}.test.tfrecord'.format(
            '_'.join(feats_fname.split('_')[:-3])))
    test_writer = tf.python_io.TFRecordWriter(test_tfrecord_name)
    for i, (img_name, cap_ids) in enumerate(
            zip(test_caps['names'], test_caps['word_to_ids'])):
        img_name = img_name.split('#')[0]
        img_feat = feats_mmap[img_to_idx[img_name], :]
        test_writer.write(
            make_example(img_feat, cap_ids, img_name).SerializeToString())
        if i % 100 == 0:
            print "test records written {}/{}".format(
                    i, len(test_caps['names']))
    test_writer.close()

    # process dev imgs and write to a record file
    dev_tfrecord_name = os.path.join(
        args.caption_tokens_dir, '{}.dev.tfrecord'.format(
            '_'.join(feats_fname.split('_')[:-3])))
    dev_writer = tf.python_io.TFRecordWriter(dev_tfrecord_name)
    for i, (img_name, cap_ids) in enumerate(
            zip(dev_caps['names'], dev_caps['word_to_ids'])):
        img_name = img_name.split('#')[0]
        img_feat = feats_mmap[img_to_idx[img_name], :]
        dev_writer.write(
            make_example(img_feat, cap_ids, img_name).SerializeToString())
        if i % 100 == 0:
            print "dev records written {}/{}".format(
                    i, len(dev_caps['names']))
    dev_writer.close()
    print "Wrote to %s" % train_tfrecord_name
    print "Wrote to %s" % dev_tfrecord_name
    print "Wrote to %s" % test_tfrecord_name


if __name__ == '__main__':
    main()

