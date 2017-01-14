from da_recon6.experiments import Experiment000
from da_recon6.utils import to_device
from da_recon6.datasets import MNISTDataReader, Separator
import numpy as np
import os
import numpy as np
import sys
import time
import chainer.functions as F
from chainer import Variable

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = 128
    inp_dim = 784
    out_dim = n_cls = 10
    n_l_train_data = 100
    n_train_data = 60000

    dims = [inp_dim, 500, 250, 100, out_dim]
    learning_rate = 1. * 1e-3
    n_epoch = 10
    decay = 0.5
    act = F.relu
    noise = True
    rc = True
    lds = True
    scale_rc = False
    scale_lds = False
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch

    # Separate dataset
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/mnist/train.npz")
    separator = Separator(n_l_train_data)
    separator.separate_then_save(fpath)

    l_train_path = os.path.join(home, "datasets/mnist/l_train.npz")
    u_train_path = os.path.join(home, "datasets/mnist/train.npz")
    test_path = os.path.join(home, "datasets/mnist/test.npz")

    # DataReader, Model, Optimizer, Losses
    data_reader = MNISTDataReader(l_train_path, u_train_path, test_path,
                                  batch_size=batch_size,
                                  n_cls=n_cls)
    exp = Experiment000(
        device,
        learning_rate,
        dims,
        act,
        noise,
        rc,
        lds,
        scale_rc,
        scale_lds,
        )

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):
        # Get data
        x_l, y_l = [Variable(to_device(x, device)) \
                        for x in data_reader.get_l_train_batch()]
        x_u, _ = [Variable(to_device(x, device)) \
                      for x in data_reader.get_u_train_batch()]

        # Train
        exp.train(x_l, y_l, x_u)
        
        # Eval
        if (i+1) % iter_epoch == 0:
            # Get data
            x_l, y_l = [Variable(to_device(x, device)) \
                            for x in data_reader.get_test_batch()]

            acc, sloss = exp.test(x_l, y_l)
            msg = "Epoch:{},ElapsedTime:{},Acc:{},SupervisedLoss:{}".format(epoch, time.time() - st, acc.data, sloss.data)
            print(msg)
            
            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    main()
