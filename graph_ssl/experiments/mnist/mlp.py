from graph_ssl.datasets import MNISTDataReader, Separator
from graph_ssl.models import GraphSSLMLPModel
from graph_ssl.datasets import to_device
import os
from chainer import optimizers
import numpy as np
import sys
import time

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = 32
    inp_dim = 784
    out_dim = n_cls = 10
    n_l_train_data = 100
    n_train_data = 60000
    n_u_train_data = n_train_data -  n_l_train_data

    dims = [inp_dim, 1000, 500, 250, 250, 250, out_dim]
    lambdas = to_device(np.array([1., 1.], np.float32), device)
    learning_rate = 1. * 1e-3
    n_epoch = 200
    iter_epoch = n_u_train_data / batch_size
    n_iter = n_epoch * iter_epoch

    # Separate dataset
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/mnist/train.npz")
    separator = Separator(n_l_train_data)
    separator.separate_then_save(fpath)

    l_train_path = os.path.join(home, "datasets/mnist/l_train.npz")
    u_train_path = os.path.join(home, "datasets/mnist/u_train.npz")
    test_path = os.path.join(home, "datasets/mnist/test.npz")

    # DataReader, Model, Optimizer
    data_reader = MNISTDataReader(l_train_path, u_train_path, test_path,
                                  batch_size=batch_size,
                                  n_cls=n_cls)
    model = GraphSSLMLPModel(dims, batch_size)
    model.to_gpu(device) if device else None
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):

        # Get data
        x_l, y_l = [to_device(x, device) for x in data_reader.get_l_train_batch()]
        x_u, _ = [to_device(x, device) for x in data_reader.get_u_train_batch()]
        x_u_0 = x_u_1 = x_u

        # Train one-step
        model.zerograds()
        loss = model(x_l, y_l, x_u_0, x_u_1)
        loss.backward()
        optimizer.update()

        # Eval
        if (i+1) % iter_epoch == 0:
            print("Evaluation at {}-th epoch".format(epoch))

            # Get data, go to test mode, eval, revert to train mode over all samples
            x_l, y_l = [to_device(x, device) for x in data_reader.get_test_batch()]
                        
            # Report
            loss = model.sloss.loss
            acc = model.sloss.accuracy
            print("Loss:{},Accuracy:{},Time/epoch:{}[s]".format(
                to_device(loss.data), to_device(acc.data) * 100, time.time() - st))
            
            epoch +=1
            st = time.time()
            
if __name__ == '__main__':
    main()
