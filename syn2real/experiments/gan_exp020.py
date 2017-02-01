from syn2real.experiments import GANExperiment
from syn2real.utils import to_device
from syn2real.datasets import MNISTDataReader, Separator
from syn2real.cnn_model import Decoder
import numpy as np
import os
import numpy as np
import sys
import time
import glob
import chainer.functions as F
from chainer import Variable
from chainer import serializers

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    model = "cnn"
    batch_size = 64
    n_l_train_data = 100
    n_train_data = 60000
    n_cls = 10
    dim_rand = 30

    learning_rate = 1. * 1e-5
    n_epoch = 100
    act = F.relu
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch

    home = os.environ.get("HOME")
    l_train_path = os.path.join(home, "datasets/mnist/l_train.npz")
    u_train_path = os.path.join(home, "datasets/mnist/train.npz")
    test_path = os.path.join(home, "datasets/mnist/test.npz")

    # DataReader, Model, Optimizer, Losses
    data_reader = MNISTDataReader(l_train_path, u_train_path, test_path,
                                  batch_size=batch_size,
                                  n_cls=n_cls)

    # Decoder
    decoder = Decoder(act=act)
    fpaths = glob.glob("./model/decoder*")
    fpaths.sort()
    fpath = fpaths[-1]
    serializers.load_hdf5(fpath, decoder)

    # Experiment
    exp = GANExperiment(
        decoder,
        device,
        model,
        dim_rand,
        n_cls,
        learning_rate,
        act,
        )

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):
        # Get data
        x_u, _ = [Variable(to_device(x, device)) \
                      for x in data_reader.get_u_train_batch()]

        # Train
        exp.train(x_u)
        
        # Eval
        if (i+1) % iter_epoch == 0:
            # Get data
            x_l, y_l = [Variable(to_device(x, device)) \
                            for x in data_reader.get_test_batch()]

            exp.test(epoch, batch_size)
            msg = "Epoch:{},ElapsedTime:{}".format(epoch, time.time() - st)
            print(msg)
            
            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    main()
