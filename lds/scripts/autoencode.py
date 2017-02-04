import numpy as np
import cv2
from chainer import serializers
from lds.cnn_model import AutoEncoder
import chainer.functions as F
import os

def main():
    # Settings
    act = F.relu

    # Model
    ae = AutoEncoder(act)

    # Load
    fpath = "/home/kzk/tmp/cnn/ae_00010.h5py"
    serializers.load_hdf5(fpath, ae)

    # Data
    home = os.environ.get("HOME")
    train_path = os.path.join(home, "datasets/mnist/train.npz")
    data = np.load(train_path)

    # Generate random vector(s)
    idx = 150
    x = data["x"][idx, :].reshape(1, 1, 28, 28).astype(np.float32)
    x = (x - 127.5) / 127.5
    y = ae.encoder(x)
    #y = F.softmax(y)

    # Generate sample(s)
    x = ae.decoder(y, test=True)
    x = x.data * 127.5 + 127.5

    cv2.imwrite("./dec_mnist_{:05d}.png".format(0), x.reshape(28, 28))

if __name__ == '__main__':
    main()
