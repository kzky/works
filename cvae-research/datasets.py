from PIL import Image
import numpy as np
import io
import os
import glob
import tarfile
import zipfile

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home

def get_np_array_from_tar_object(tar_extractfl):
    '''converts a buffer from a tar file in np.array'''
    return np.asarray(
        bytearray(tar_extractfl.read()), 
        dtype=np.uint8)


def data_iterator_celebA(img_path, batch_size=64, ih=128, iw=128, 
                         shuffle=True, rng=None):
    imgs += glob.glob("{}/*.png".format(img_path))

    def load_func(i):
        img = Image.open(imgs[i])
        img = np.asarray(img).transpose(2, 0, 1)
        cx = 89
        cy = 121
        img = img[cy - 64: cy + 64, 
                  cx - 64: cx + 64, :] / 255.
        img = img * 2. - 1.
        return img, None


    return data_iterator_simple(
        load_func, len(imgs), batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


if __name__ == '__main__':
