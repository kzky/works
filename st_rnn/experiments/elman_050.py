from st_rss.models import ElmanOnestep, ElmanNet, RNNLabeledLosses, RNNUnlabeledLosses
from st_rss.datasets import MNISTDataReader, Separator
from st_rss.utils import to_device
from st_rss.elmans import forward_backward_update_050, evaluate_050
import numpy as np
import os
from chainer import optimizers
import numpy as np
import sys
import time
import chainer.functions as F

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    T = 5
    batch_size = 128
    inp_dim = 784
    out_dim = n_cls = 10
    n_l_train_data = 100
    n_train_data = 60000

    dims = [inp_dim, 250, 100, out_dim]
    lambdas = to_device(np.array([1., 1.], np.float32), device)
    learning_rate = 1. * 1e-3
    n_epoch = 100
    decay = 0.5
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
                                  n_cls=n_cls)
    model = ElmanOnestep(dims)
    rnn = ElmanNet(onestep, T)
    model.to_gpu(device) if device else None
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)
    rnn_labeled_loss = RNNLabeledLosses()
    rnn_unlabeled_loss = RNNUnlabeledLosses()
    
    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):

        # Get data
        x_l, y_l = [to_device(x, device) for x in data_reader.get_l_train_batch()]
        x_u, _ = [to_device(x, device) for x in data_reader.get_u_train_batch()]

        # Forward/Backward
        forward_backward_update_050(
            rnn, rnn_labeled_loss, rnn_unlabeled_loss,
            optimizer, model,
            x_l, y_l, x_u)
        
        # Eval
        if (i+1) % iter_epoch == 0:
            print("Evaluation at {}-th epoch".format(epoch))

            # Get data
            x_l, y_l = [to_device(x, device) for x in data_reader.get_test_batch()]

            # Compute loss and accuracy
            losses = evaluate_050(rnn, rnn_labeled_loss, x_l, y_l)
            
            # Report
            print("Loss:{},Accuracy:{},Time/epoch:{}[s]".format(
                [to_device(loss.loss) for loss in losses],
                [to_device(loss.accuracy) * 100 for loss in losses],
                time.time() - st))
            
            epoch +=1
            st = time.time()
            
if __name__ == '__main__':
    main()
