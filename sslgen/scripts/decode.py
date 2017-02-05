import numpy as np
import cv2
from chainer import serializers
from sslgen.cnn_model import Decoder
import chainer.functions as F

def main():
    # Settings
    act = F.relu

    # Model
    decoder = Decoder(act)

    # Load
    fpath = "/home/kzk/tmp/cnn/decoder_00010.h5py"
    serializers.load_hdf5(fpath, decoder)

    # Generate random vector(s)
    bs = 1
    y = np.random.rand(bs, 10).astype(np.float32)
    y = np.array([[1000.0, .0, 0.0, 0.0, 0.0, 100, 0.0, 0.0, 0.0, 100.0]], dtype=np.float32)
    y = y / np.sum(y)
    y = y[np.newaxis]
    # Generate sample(s)
    x = decoder(y, test=True)
    x = x.data 
    cv2.imwrite("./dec_mnist.png", x.reshape(28, 28) * 127.5 + 127.5)

if __name__ == '__main__':
    main()
