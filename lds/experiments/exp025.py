from lds.experiments import Experiment025
from lds.utils import to_device
from lds.datasets import MNISTDataReader, Separator
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
    n_l_train_data = 100
    n_train_data = 60000
    n_cls = 10

    learning_rate = 1. * 1e-3
    n_epoch = 50
    act = F.relu
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
                                  n_cls=n_cls,
                                  da=True,
                                  shape=True)
    exp = Experiment025(
        device,
        learning_rate,
        act,
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
        acc_prev = 0.
        if (i+1) % iter_epoch == 0:
            # Get data
            x_l, y_l = [Variable(to_device(x, device)) \
                            for x in data_reader.get_test_batch()]

            accs = [float(acc.data) for acc in exp.test(x_l, y_l)]
            msg = "Epoch:{},ElapsedTime:{},Acc:{}".format(
                epoch,
                time.time() - st, 
                "|".join(map(str, accs)))
            print(msg)
            if acc_prev > accs[-1]:
                exp.lambda_ne *= 0.5
            acc_prev = accs[-1]

            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    main()
