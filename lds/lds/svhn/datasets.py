import numpy as np
import scipy.io
import os
from chainer import cuda
import cv2

class SVHNDataReader(object):
    """DataReader
    """
    def __init__(
            self,
            l_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/svhn/l_train_32x32.mat",
            u_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/svhn/u_train_32x32.mat", 
            test_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/svhn/test_32x32.mat",
            batch_size=64,
            n_cls=10,
            da=False,
            shape=False,
    ):
            
        _l_train_data = dict(scipy.io.loadmat(l_train_path))
        self.l_train_path = {
            "X":_l_train_data["X"], 
            "y":np.squeeze(_l_train_data["y"])}
        _u_train_data = scipy.io.loadmat(u_train_path)
        self.u_train_data = {
            "X":_u_train_data["X"], 
            "y":np.squeeze(_u_train_data["y"])}
        _test_data = scipy.io.loadmat(test_path)
        self.test_data = {
            "X": _test_data["X"].transpose((3, 2, 0, 1)), 
            "y": np.squeeze(_test_data["y"])}

        self._batch_size = batch_size
        self._next_position_l_train = 0
        self._next_position_u_train = 0

        self._n_l_train_data = len(self.l_train_data["X"])
        self._n_u_train_data = len(self.u_train_data["X"])
        self._n_test_data = len(self.test_data["X"])
        self._n_cls = n_cls
        self._da = da
        self._shape = shape

        print("Num. of labeled samples {}".format(self._n_l_train_data))
        print("Num. of unlabeled samples {}".format(self._n_u_train_data))
        print("Num. of test samples {}".format(self._n_test_data))
        print("Num. of classes {}".format(self._n_cls))

    def get_l_train_batch(self,):
        """Return next batch data.

        Return next batch data. Once all samples are read, permutate all samples.
        
        Returns
        ------------
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """
        # Read data
        beg = self._next_position_l_train
        end = self._next_position_l_train+self._batch_size
        batch_data_x_ = self.l_train_data["X"][beg:end, :]
        batch_data_y_ = self.l_train_data["y"][beg:end]
        batch_data_x = ((batch_data_x_ - 127.5)/ 127.5).astype(np.float32)
        if self._da:
            batch_data_x = self._transform(batch_data_x)
        batch_data_y = batch_data_y_.astype(np.int32)

        # Reset pointer
        self._next_position_l_train += self._batch_size
        if self._next_position_l_train >= self._n_l_train_data:
            self._next_position_l_train = 0

            # Shuffle
            idx = np.arange(self._n_l_train_data)
            np.random.shuffle(idx)
            self.l_train_data["X"] = self.l_train_data["X"][idx]
            self.l_train_data["y"] = self.l_train_data["y"][idx]

        return batch_data_x, batch_data_y

    def get_u_train_batch(self,):
        """Return next batch data.

        Return next batch data. Once all samples are read, permutate all samples.

        Returns:
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """
        # Read data
        beg = self._next_position_u_train
        end = self._next_position_u_train+self._batch_size
        batch_data_x_ = self.u_train_data["X"][beg:end, :]
        batch_data_y_ = self.u_train_data["y"][beg:end]
        batch_data_x = ((batch_data_x_ - 127.5)/ 127.5).astype(np.float32)
        if self._da:
            batch_data_x = self._transform(batch_data_x)
        batch_data_y = batch_data_y_.astype(np.int32)

        # Reset pointer
        self._next_position_u_train += self._batch_size
        if self._next_position_u_train >= self._n_u_train_data:
            self._next_position_u_train = 0

            # Shuffle
            idx = np.arange(self._n_u_train_data)
            np.random.shuffle(idx)
            self.u_train_data["X"] = self.u_train_data["X"][idx]
            self.u_train_data["y"] = self.u_train_data["y"][idx]

        return batch_data_x, batch_data_y

    def get_test_batch(self,):
        """Return next batch data.

        Returns
        ------------
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """

        # Read data
        batch_data_x_ = self.test_data["X"]
        batch_data_y_ = self.test_data["y"]
        batch_data_x = ((batch_data_x_ - 127.5)/ 127.5).astype(np.float32)
        batch_data_y = batch_data_y_.astype(np.int32)

        return batch_data_x , batch_data_y

    def _transform(self, imgs):
        #TODO
        return imgs

class Separator(object):
    """Seprate the original samples to labeled samples and unlabeled samples.

    Seprate the original samples to labeled samples and unlabeled samples in such
    way; the number of labeled samples are selected randomly, it is equal to `l`, 
    and the others are unlabeled samples.

    Note SVNH dataset is of shape (h, w, c, b) so that Separaor transform to 
    the shape (b, c, h, w) for the later use efficiently.
    """

    def __init__(self, l=1000):
        self.l = l

    def separate_then_save(
            self,
            fpath="/home/kzk/.chainer/dataset/pfnet/chainer/svhn/train_32x32.mat"):
        ldata, udata = self._separate(fpath)
        self._save_ssl_data(fpath, ldata, udata)
        
    def _separate(
            self,
            fpath="/home/kzk/.chainer/dataset/pfnet/chainer/svhn/train_32x32.mat"):
        
        data = scipy.io.loadmat(fpath)
        n = data["X"].shape[-1]
        idxs = np.arange(n)
        idxs_l = self._sample_indices(np.squeeze(data["y"]))
        idxs_u = np.asarray(list(set(idxs) - set(idxs_l)))

        ldata = {}
        udata = {}
        ldata["X"] = data["X"][:, :, :, idxs_l].transpose((3, 2, 0, 1))
        ldata["y"] = np.squeeze(data["y"])[idxs_l]
        udata["X"] = data["X"][:, :, :, idxs_u].transpose((3, 2, 0, 1))
        udata["y"] = np.squeeze(data["y"])[idxs_u]

        # Shuffle in advance since svhn label is ordered sequencially like 0, 0, ..., 1, 1, ..
        idx = np.arange(len(ldata["X"]))
        np.random.shuffle(idx)
        ldata["X"] = ldata["X"][idx]
        ldata["y"] = ldata["y"][idx]
        
        idx = np.arange(len(udata["X"]))
        np.random.shuffle(idx)
        udata["X"] = udata["X"][idx]
        udata["y"] = udata["y"][idx]
        
        return ldata, udata

    def _sample_indices(self, y):
        classes = set(np.squeeze(y))
        indicies = []
        n_for_each_classes = int(1. * self.l / len(classes))
        for c in classes:
            indices_for_c = np.where(y==c)[0]
            indicies += np.random.choice(indices_for_c, n_for_each_classes,
                                            replace=False).tolist()
        return indicies
        
    def _save_ssl_data(self, fpath, ldata, udata):
        dpath = os.path.dirname(fpath)
        fname = os.path.basename(fpath)

        l_fname = "l_{}".format(fname)
        u_fname = "u_{}".format(fname)
        
        ldata_fpath = os.path.join(dpath, l_fname)
        udata_fpath = os.path.join(dpath, u_fname)

        scipy.io.savemat(ldata_fpath, ldata)
        scipy.io.savemat(udata_fpath, udata)
