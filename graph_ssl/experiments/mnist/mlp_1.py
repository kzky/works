from graph_ssl.datasets import MNISTDataReader, Separator
from graph_ssl.models_1 import GraphSSLMLPModel
from graph_ssl.utils import to_device
import os
from chainer import optimizers
import numpy as np
import sys
import time
import chainer.functions as F

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = 100
    inp_dim = 784
    out_dim = n_cls = 10
    n_l_train_data = 100
    n_train_data = 60000

    dims = [inp_dim, 1000, 500, 250, 250, 250, out_dim]
    lambdas = to_device(np.array([1., 1.], np.float32), device)
    learning_rate = 1. * 1e-3
    n_epoch = 200
    decay = 0.5
    act = F.tanh
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

    # DataReader, Model, Optimizer
    data_reader = MNISTDataReader(l_train_path, u_train_path, test_path,
                                  batch_size=batch_size,
                                  n_cls=n_cls)
    model = GraphSSLMLPModel(dims, batch_size, act, decay, lambdas, device)
    model.to_gpu(device) if device else None
    optimizer = optimizers.Adam(learning_rate)
    optimizer.use_cleargrads()
    optimizer.setup(model)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):

        # Get data
        x_l, y_l = [to_device(x, device) for x in data_reader.get_l_train_batch()]
        x_u, _ = [to_device(x, device) for x in data_reader.get_u_train_batch()]
        y_l_float32 = np.zeros((n_l_train_data, n_cls), dtype=np.float32)
        y_l_float32[np.arange(n_l_train_data), y_l]  = 1.
        y_l_float32 = to_device(y_l_float32)
        
        # Train one-step
        model.zerograds()
        loss = model(x_l, y_l, x_u, y_l_float32)
        loss.backward()
        optimizer.update()

        # Eval
        if (i+1) % iter_epoch == 0:
            print("Evaluation at {}-th epoch".format(epoch))

            # Get data, go to test mode, eval, revert to train mode over all samples
            x_l, y_l = [to_device(x, device) for x in data_reader.get_test_batch()]
            model.classifier.test = True
            model.sloss(x_l, y_l)
            model.classifier.test = False
            
            # Report
            sloss = model.sloss.loss
            gloss = model.gloss.loss
            acc = model.sloss.accuracy
            print("SLoss:{},GLoss:{},Accuracy:{},Time/epoch:{}[s]".format(
                to_device(sloss.data), to_device(gloss.data),
                to_device(acc.data) * 100, time.time() - st))
            for p, y in zip(to_device(model.sloss.pred.data), y_l):
                print(p)
                print(y)
            epoch +=1
            st = time.time()
            
if __name__ == '__main__':
    main()
