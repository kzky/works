import numpy as np
import os
import cv2

class MNISTDataReader(object):
    """DataReader
    """
    def __init__(
            self,
            l_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/l_mnist.npz",
            u_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/mnist.npz", 
            test_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/mnist.npz",
            batch_size=64,
            n_cls=10,
            da=False,
            shape=False,
    ):
        # Load dataset
        self.l_train_data = dict(np.load(l_train_path))
        self.l_train_data = {
            "train_x":self.l_train_data["train_x"], 
            "train_y":self.l_train_data["train_y"][:, np.newaxis]}
        _u_train_data = np.load(u_train_path)
        self.u_train_data = {
            "train_x":_u_train_data["train_x"], 
            "train_y":_u_train_data["train_y"][:, np.newaxis]}
        _test_data = np.load(test_path)
        self.test_data = {
            "test_x": _test_data["x"], 
            "test_y": _test_data["y"][:, np.newaxis]}

        self._batch_size = batch_size
        self._next_position_l_train = 0
        self._next_position_u_train = 0

        self._n_l_train_data = len(self.l_train_data["train_x"])
        self._n_u_train_data = len(self.u_train_data["train_x"])
        self._n_test_data = len(self.test_data["test_x"])
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
            return x.reshape(bs, 1, 28, 28)
        return x
    
    def get_l_train_batch(self, batch_size=None):
        """Return next batch data.

        Return next batch data. Once all samples are read, permutate all samples.
        
        Returns
        ------------
        tuple of 3: ndarray (data), placeholder, ndarray (label)

        """
        # Read data
        if batch_size is None:
            _batch_size = self._batch_size
        else:
            _batch_size = batch_size

        beg = self._next_position_l_train
        end = self._next_position_l_train + _batch_size
        if end < self._n_l_train_data:
            batch_data_x_ = self.l_train_data["train_x"][beg:end, :]
            batch_data_y_ = self.l_train_data["train_y"][beg:end]
            batch_data_x = (batch_data_x_/ 255.).astype(np.float32)
        else:
            bs_s = _batch_size - self._n_l_train_data + beg
            batch_data_x_ = self.l_train_data["train_x"][beg:end, :]
            batch_data_y_ = self.l_train_data["train_y"][beg:end]
            batch_data_x__ = self.l_train_data["train_x"][0:bs_s, :]
            batch_data_y__ = self.l_train_data["train_y"][0:bs_s]
            batch_data_x_ = np.concatenate(
                (batch_data_x_, batch_data_x__))
            batch_data_y_ = np.concatenate(
                (batch_data_y_, batch_data_y__))
            batch_data_x = (batch_data_x_/ 255.).astype(np.float32)

        batch_data_x0 = self._transform(batch_data_x)
        batch_data_y = batch_data_y_.astype(np.int32)

        # Reset pointer
        self._next_position_l_train += _batch_size
        if self._next_position_l_train >= self._n_l_train_data:
            self._next_position_l_train = 0

            # Shuffle
            idx = np.arange(self._n_l_train_data)
            np.random.shuffle(idx)
            self.l_train_data["train_x"] = self.l_train_data["train_x"][idx]
            self.l_train_data["train_y"] = self.l_train_data["train_y"][idx]

        batch_data_x0 = self.reshape(batch_data_x0)
        return batch_data_x0, None, batch_data_y

    def get_u_train_batch(self, batch_size=None):
        """Return next batch data.

        Return next batch data. Once all samples are read, permutate all samples.

        Returns:
        tuple of 3: ndarray (data), ndarray (data), ndarray (label)

        """
        # Read data
        if batch_size is None:
            _batch_size = self._batch_size
        else:
            _batch_size = batch_size

        beg = self._next_position_u_train
        end = self._next_position_u_train + _batch_size
        if end < self._n_u_train_data:
            batch_data_x_ = self.u_train_data["train_x"][beg:end, :]
            batch_data_y_ = self.u_train_data["train_y"][beg:end]
            batch_data_x = (batch_data_x_/ 255.).astype(np.float32)
        else:
            bs_s = _batch_size - self._n_u_train_data + beg
            batch_data_x_ = self.u_train_data["train_x"][beg:end, :]
            batch_data_y_ = self.u_train_data["train_y"][beg:end]
            batch_data_x__ = self.u_train_data["train_x"][0:bs_s, :]
            batch_data_y__ = self.u_train_data["train_y"][0:bs_s]
            batch_data_x_ = np.concatenate(
                (batch_data_x_, batch_data_x__))
            batch_data_y_ = np.concatenate(
                (batch_data_y_, batch_data_y__))
            batch_data_x = (batch_data_x_/ 255.).astype(np.float32)

        batch_data_x0 = self._transform(batch_data_x)
        batch_data_x1 = self._transform(batch_data_x)
        batch_data_y = batch_data_y_.astype(np.int32)

        # Reset pointer
        self._next_position_u_train += _batch_size
        if self._next_position_u_train >= self._n_u_train_data:
            self._next_position_u_train = 0

            # Shuffle
            idx = np.arange(self._n_u_train_data)
            np.random.shuffle(idx)
            self.u_train_data["train_x"] = self.u_train_data["train_x"][idx]
            self.u_train_data["train_y"] = self.u_train_data["train_y"][idx]
 
        batch_data_x0 = self.reshape(batch_data_x0)
        batch_data_x1 = self.reshape(batch_data_x1)
        return batch_data_x0, batch_data_x1, batch_data_y

    def get_test_batch(self,):
        """Return next batch data.

        Returns
        ------------
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """

        # Read data
        batch_data_x_ = self.test_data["test_x"]
        batch_data_y_ = self.test_data["test_y"]
        batch_data_x = (batch_data_x_ / 255.).astype(np.float32)
        batch_data_y = batch_data_y_.astype(np.int32)

        batch_data_x = self.reshape(batch_data_x)
        return batch_data_x , batch_data_y

    def _transform(self, imgs):
        bs = imgs.shape[0]
        imgs = imgs.reshape(bs, 1, 28, 28)
        imgs_ = np.zeros_like(imgs)
        for i, img in enumerate(imgs):
            # random flip
            if np.random.randint(2):
                img_ = np.copy(img[:, :, ::-1])
            else:
                img_ = np.copy(img)

            # rotation
            n = np.random.choice(np.arange(-15, 15))
            M = cv2.getRotationMatrix2D((28/2, 28/2), n, 1)
            dst = cv2.warpAffine(img_.transpose(1, 2, 0), M, (28, 28))

            # translation
            M = np.float32([[1,0,np.random.randint(-2, 2)],
                            [0,1,np.random.randint(-2, 2)]])
            dst = cv2.warpAffine(dst, M, (28, 28))
            imgs_[i] = dst.reshape((1, 28, 28))

        imgs_ = imgs_.reshape(bs, 28*28)
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
            fpath="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/mnist.npz"):
        ldata, udata = self._separate(fpath)
        self._save_ssl_data(fpath, ldata, udata)
        
    def _separate(
            self,
            fpath="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/mnist.npz"):
        
        data = np.load(fpath)
        n = len(data["x"])
        idxs = np.arange(n)
        idxs_l = self._sample_indices(data["y"])
        idxs_u = np.asarray(list(set(idxs) - set(idxs_l)))

        ldata = {}
        udata = {}
        ldata["train_x"] = data["x"][idxs_l]
        ldata["train_y"] = data["y"][idxs_l]
        udata["train_x"] = data["x"][idxs_u]
        udata["train_y"] = data["y"][idxs_u]

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
