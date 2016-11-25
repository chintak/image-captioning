from os.path import join
from pandas import read_csv


caps_per_name = 5

train_names_file = '../Flickr8k_Captions/Flickr_8k.trainImages.txt'
test_names_file = '../Flickr8k_Captions/Flickr_8k.testImages.txt'
dev_names_file = '../Flickr8k_Captions/Flickr_8k.devImages.txt'

caption_file = '../Flickr8k_Captions/Flickr8k.token.txt'
out_file_pre = '../Flickr8k_Captions/Flickr8k.token.'

# read the train img ids
with open(train_names_file, 'r') as fp:
    train_ids = ["{}#{}".format(n.strip(), i) for n in fp.readlines()
                 for i in range(5)]

# read the test img ids
with open(test_names_file, 'r') as fp:
    test_ids = ["{}#{}".format(n.strip(), i) for n in fp.readlines()
                for i in range(5)]

# read the dev img ids
with open(dev_names_file, 'r') as fp:
    dev_ids = ["{}#{}".format(n.strip(), i) for n in fp.readlines()
               for i in range(5)]

df = read_csv(caption_file, sep='\t', names=['name', 'text'])
df.set_index('name', inplace=True)

# save the captions for train, test and dev images in separate files
df.ix[train_ids].to_csv("{}trainImgs.txt".format(out_file_pre), sep='\t',
                        header=False)
df.ix[test_ids].to_csv("{}testImgs.txt".format(out_file_pre), sep='\t',
                       header=False)
df.ix[dev_ids].to_csv("{}devImgs.txt".format(out_file_pre), sep='\t',
                      header=False)

