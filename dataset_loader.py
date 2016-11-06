import os
import glob
import logging
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
from joblib import Parallel, delayed
from pprint import pformat
from utils import CONFIG

config = CONFIG.DatasetLoader
log = logging.getLogger('DatasetLoader')
log.setLevel(config.logLevel)


class DatasetLoader(object):
    """docstring for DatasetLoader"""

    def __init__(self, folder_name, ext='jpg', size=224, batch_size=None):
        self.folder_name = folder_name
        self.input_size = size
        self._par = Parallel(n_jobs=config.par_jobs)
        filelist = "{}_list.txt".format(os.path.abspath(folder_name))
        if os.path.exists(filelist):
            with open(filelist, 'r') as fp:
                self._fnames = [f.strip() for f in fp.readlines()]
        else:
            self._fnames = glob.glob(os.path.join(folder_name, '*.%s' % ext))
            with open(filelist, 'w') as fp:
                fp.write("\n".join(self._fnames))
        self.num_samples = len(self._fnames)
        self.curr_img_idx = 0
        self._batch_size = batch_size
        self.passes = 0
        log.info("Number of images found: %d" % self.num_samples)

    def get_batch(self, batch_size=None, warp=True):
        batch_size = self._batch_size if batch_size is None else batch_size
        assert batch_size is not None
        if self.curr_img_idx + batch_size < self.num_samples:
            names = self._fnames[
                self.curr_img_idx: self.curr_img_idx + batch_size]
            assert len(names) == batch_size
            self.curr_img_idx += batch_size
        else:
            remains = self.num_samples - self.curr_img_idx
            warp_count = 0
            if warp:
                warp_count = batch_size - remains
                names = self._fnames[-remains:] + self._fnames[:warp_count]
                assert len(names) == batch_size
            else:
                names = self._fnames[-remains:]
                log.debug("num of files {}".format(len(names)))
            self.curr_img_idx = warp_count
            self.passes += 1
        X = self._par(delayed(load_image)(path, self.input_size)
                      for path in names)
        X = np.asarray(X, dtype=np.float32)
        # X = np.transpose(X, (1, 2, 3, 0))
        log.debug("size of batch fed {}".format(X.shape))
        return X


def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    if resized_img.ndim == 2:
        resized_img = skimage.color.gray2rgb(resized_img)
    assert resized_img.ndim == 3, ("RGB image expected")
    return resized_img
