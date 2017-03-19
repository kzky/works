import numpy as np
import os
from chainer import cuda
import cv2

class Cifar10DataReader(object):
    """DataReader
    """
    def __init__(
            self,
            l_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/cifar10/l_cifar-10.npz",
            u_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/cifar10/cifar-10.npz", 
            test_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/cifar10/cifar-10.npz",
            zca_path=None,
            batch_size=64,
            n_cls=10,
            da=False,
            shape=False,
    ):
        # Load dataset
        self.l_train_data = dict(np.load(l_train_path))
        _u_train_data = np.load(u_train_path)
        self.u_train_data = {
            "train_x":_u_train_data["train_x"], 
            "train_y":_u_train_data["train_y"]}
        _test_data = np.load(test_path)
        self.test_data = {
            "test_x": _test_data["test_x"], 
            "test_y": _test_data["test_y"]}

        # ZCA Whitening
        if zca_path is not None:
            zca_comp = np.load(zca_path)
            X_mean = zca_comp["X_mean"]
            W_zca = zca_comp["W_zca"]
            
            self.l_train_data = {
                "train_x": np.dot(self.l_train_data["train_x"] - X_mean, W_zca.T),
                "train_y": self.l_train_data["train_y"]}
            self.u_train_data = {
                "train_x":np.dot(self.u_train_data["train_x"] - X_mean, W_zca.T), 
                "train_y":self.u_train_data["train_y"]}
            _test_data = np.load(test_path)
            self.test_data = {
                "test_x": np.dot(self.test_data["test_x"] - X_mean, W_zca.T), 
                "test_y": self.test_data["test_y"]}

        self._batch_size = batch_size
        self._next_position_l_train = 0
        self._next_position_u_train = 0

        self._n_l_train_data = len(self.l_train_data["train_x"])
        self._n_u_train_data = len(self.u_train_data["train_x"])
        self._n_test_data = len(self.test_data["train_x"])
        self._n_cls = n_cls
        self._da = da
        self._shape = shape

        print("Num. of labeled samples {}".format(self._n_l_train_data))
        print("Num. of unlabeled samples {}".format(self._n_u_train_data))
        print("Num. of test samples {}".format(self._n_test_data))
        print("Num. of classes {}".format(self._n_cls))

    def reshape(self, x):
        if self._shape:
            bs = x.shape[0]
            return x.reshape(bs, 3, 32, 32)
        return x
    
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
        batch_data_x_ = self.l_train_data["train_x"][beg:end, :]
        batch_data_y_ = self.l_train_data["train_y"][beg:end]
        batch_data_x = (batch_data_x_/ 255.).astype(np.float32)
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
            self.l_train_data["train_x"] = self.l_train_data["train_x"][idx]
            self.l_train_data["train_y"] = self.l_train_data["train_y"][idx]

        batch_data_x = self.reshape(batch_data_x)
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
        batch_data_x_ = self.u_train_data["train_x"][beg:end, :]
        batch_data_y_ = self.u_train_data["train_y"][beg:end]
        batch_data_x = (batch_data_x_ / 255.).astype(np.float32)
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
            self.u_train_data["train_x"] = self.u_train_data["train_x"][idx]
            self.u_train_data["train_y"] = self.u_train_data["train_y"][idx]
 
        batch_data_x = self.reshape(batch_data_x)
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
        batch_data_x_ = self.test_data["train_x"]
        batch_data_y_ = self.test_data["train_y"]
        batch_data_x = (batch_data_x_ / 255.).astype(np.float32)
        batch_data_y = batch_data_y_.astype(np.int32)

        batch_data_x = self.reshape(batch_data_x)
        return batch_data_x , batch_data_y

    def _transform(self, imgs):
        bs = imgs.shape[0]
        imgs = imgs.reshape(bs, 3, 32, 32)
        imgs_ = np.zeros_like(imgs)
        for i, img in enumerate(imgs):
            # random flip
            if np.random.randint(2):
                imgs_[i] = img[:, :, ::-1]
            else:
                imgs_[i] = img
        imgs_ = imgs_.reshape(bs, 3072)
        return imgs_

class Separator(object):
    """Seprate the original samples to labeled samples and unlabeled samples.

    Seprate the original samples to labeled samples and unlabeled samples in such
    way; the number of labeled samples are selected randomly, it is equal to `l`, 
    and the others are unlabeled samples.
    """

    def __init__(self, l=4000):
        self.l = l

    def separate_then_save(
            self,
            fpath="/home/kzk/.chainer/dataset/pfnet/chainer/cifar10/cifar-10.npz"):
        ldata, udata = self._separate(fpath)
        self._save_ssl_data(fpath, ldata, udata)
        
    def _separate(
            self,
            fpath="/home/kzk/.chainer/dataset/pfnet/chainer/cifar10/cifar-10.npz"):
        
        data = np.load(fpath)
        n = len(data["train_x"])
        idxs = np.arange(n)
        idxs_l = self._sample_indices(data["train_y"])
        idxs_u = np.asarray(list(set(idxs) - set(idxs_l)))

        ldata = {}
        udata = {}
        ldata["train_x"] = data["train_x"][idxs_l]
        ldata["train_y"] = data["train_y"][idxs_l]
        udata["train_x"] = data["train_x"][idxs_u]
        udata["train_y"] = data["train_y"][idxs_u]

        return ldata, udata

    def _sample_indices(self, y):
        classes = set(y)
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

        np.savez(ldata_fpath, **ldata)
        np.savez(udata_fpath, **udata)
