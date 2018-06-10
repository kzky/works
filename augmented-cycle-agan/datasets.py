"""
Provide data iterator for horse2zebra examples.
"""

import os
import scipy.misc
import zipfile
from contextlib import contextmanager
import numpy as np
import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home

import cv2
import tarfile

def load_pix2pix_dataset(dataset="edges2shoes", train=True, 
                         normalize_method=lambda x: (x - 127.5) / 127.5):
    image_uri = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz'\
                .format(dataset)
    logger.info('Getting {} data from {}.'.format(dataset, image_uri))
    r = download(image_uri)

    # Load concatenated images, then save separately
    #TODO: how to do for test
    img_A_list = []
    img_B_list = []

    with tarfile.open(fileobj=r, mode="r") as tar:
        for tinfo in tar.getmembers():
            print(tinfo)
            if not ".jpg" in tinfo.name:
                continue
            f = tar.extractfile(tinfo)
            img = scipy.misc.imread(f, mode="RGB")
            #img = cv2.imread(f)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            img_A = img[:, 0:w // 2, :].transpose((2, 0, 1))
            img_B = img[:, w // 2:, :].transpose((2, 0, 1))
            img_A_list.append(img_A)
            img_B_list.append(img_B)

            # # Save the loaded image eparately
            # dataset_name, train_or_valid, img_name = tinfo.name.split("/")
            # save_fpath_A = os.path.join(get_data_home(), dataset_name, 
            #                             "{}_A".format(train_or_valid), img_name)
            # save_fpath_B = os.path.join(get_data_home(), dataset_name, 
            #                             "{}_B".format(train_or_valid), img_name)
            # scipy.misc.imsave(save_fpath_A, img_A.transpose(1, 2, 0))
            # scipy.misc.imsave(save_fpath_B, img_B.transpose(1, 2, 0))

    r.close()
    logger.info('Getting image data done.')
    return np.asarray(img_A_list), np.asarray(img_B_list)

#TODO: abstract data source under the autmented-cycle gan dataset
class Edges2ShoesDataSource(DataSource):
    
    def _get_data(self, position):
        if self._paired:
            images_A = self._images_A[self._indices[position]]
            images_B = self._images_B[self._indices[position]]
        else:
            images_A = self._images_A[self._indices_A[position]]
            images_B = self._images_B[self._indices_B[position]]
        return images_A, images_A

    def __init__(self, dataset="edges2shoes", train=True, paired=True, shuffle=False, rng=None):
        super(Edges2ShoesDataSource, self).__init__(shuffle=shuffle)

        if rng is None:
            rng = np.random.RandomState(313)

        images_A, images_B = load_pix2pix_dataset(dataset=dataset, train=train)
        self._images_A = images_A
        self._images_B = images_B
        self._size = len(self._images_A)   # since A and B is the paired image, lengths are the same
        self._size_A = len(self._images_A)
        self._size_B = len(self._images_B)
        self._variables = ('A', 'B')
        self._paired = paired
        self.rng = rng
        self.reset()

    def reset(self):
        if self._paired:
            self._indices = self.rng.permutation(self._size) \
                            if self._shuffle else np.arange(self._size)
                               
        else:
            self._indices_A = self.rng.permutation(self._size_A) \
                              if self._shuffle else np.arange(self._size_A)
            self._indices_B = self.rng.permutation(self._size_B) \
                              if self._shuffle else np.arange(self._size_B)
        return super(Edges2ShoesDataSource, self).reset()
    

def pix2pix_data_source(dataset, train=True, paired=True, shuffle=False, rng=None):
    return Pix2PixDataSource(dataset=dataset,
                             train=train, paired=paired, shuffle=shuffle, rng=rng)

def pi2pix_data_iterator(data_source, batch_size):
    return data_iterator(data_source,
                         batch_size=batch_size,
                         with_memory_cache=False,
                         with_file_cache=False)

if __name__ == '__main__':
    # Hand-made test
    
    dataset = "edges2shoes"
    ds = Edges2ShoesDataSource(dataset=dataset)
    di = pi2pix_data_iterator(ds, batch_size=1)
    

    
    
