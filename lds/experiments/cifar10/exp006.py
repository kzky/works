from lds.cifar10.experiments import Experiment006
from lds.utils import to_device
from lds.cifar10.datasets import Cifar10DataReader, Separator
import numpy as np
import os
import numpy as np
import sys
import time
import chainer.functions as F
from chainer import Variable
from chainer import cuda

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = 128
    n_l_train_data = 4000
    n_train_data = 50000
    n_cls = 10

    learning_rate = 1. * 1e-3
    n_epoch = 50
    act = F.relu
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch

    # Separate dataset
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/cifar10/cifar-10.npz")
    separator = Separator(n_l_train_data)
    separator.separate_then_save(fpath)

    l_train_path = os.path.join(home, "datasets/cifar10/l_cifar-10.npz")
    u_train_path = os.path.join(home, "datasets/cifar10/cifar-10.npz")
    test_path = os.path.join(home, "datasets/cifar10/cifar-10.npz")
    zca_path = os.path.join(home, "datasets/cifar10/zca_components.npz")

    # DataReader, Model, Optimizer, Losses
    data_reader = Cifar10DataReader(l_train_path, u_train_path, test_path,
                                    zca_path=zca_path, 
                                    batch_size=batch_size,
                                    n_cls=n_cls,
                                    da=True,
                                    shape=True)
    exp = Experiment006(
        device,
        learning_rate,
        act,
        )

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    acc_prev = 0.
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
            bs = 100
            accs = []
            for i in range(0, x_l.shape[0], bs):
                acc = exp.test(x_l[i:i+bs, ], y_l[i:i+bs, ])
                accs.append(cuda.to_cpu(acc.data))
            acc_mean = np.mean(accs)
            msg = "Epoch:{},ElapsedTime:{},Acc:{}".format(
                epoch,
                time.time() - st, 
                acc_mean)
            print(msg)
            if acc_prev > acc_mean:
                exp.lambda_ *= 0.5
                print("lambda decay")
            acc_prev = acc_mean

            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    main()
