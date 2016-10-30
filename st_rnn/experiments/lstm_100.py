from st_rnn.models import LSTMOnestep, LSTMNet, RNNLabeledLosses, RNNUnlabeledLosses
from st_rnn.datasets import MNISTDataReader, Separator
from st_rnn.utils import to_device
from st_rnn.lstms import forward_backward_update_100, evaluate_100
import numpy as np
import os
from chainer import optimizers
import numpy as np
import sys
import time
import chainer.functions as F
import chainer

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
    l_type = "hard"

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
    model = LSTMOnestep(dims)
    rnn = LSTMNet(model, T)
    model.to_gpu(device) if device else None

    #TODO: Add gradient clip as hook
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.GradientHardClipping(-3, 3))
    rnn_labeled_losses = RNNLabeledLosses(T)
    rnn_unlabeled_losses = RNNUnlabeledLosses(T, l_type)
    
    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):
        # Get data
        x_l, y_l = [to_device(x, device) for x in data_reader.get_l_train_batch()]
        x_u, _ = [to_device(x, device) for x in data_reader.get_u_train_batch()]

        # Forward/Backward
        forward_backward_update_100(
            rnn, rnn_labeled_losses, rnn_unlabeled_losses,
            optimizer, model,
            x_l, y_l, x_u)
        
        # Eval
        if (i+1) % iter_epoch == 0:
            print("Evaluation at {}-th epoch".format(epoch))

            # Get data
            x_l, y_l = [to_device(x, device) for x in data_reader.get_test_batch()]

            # Compute loss and accuracy
            losses = evaluate_100(rnn, rnn_labeled_losses, model,  x_l, y_l)
            
            # Report
            print("Loss:{},Accuracy:{},Time/epoch:{}[s]".format(
                [float(to_device(loss.data, device)) for loss in losses],
                [float(to_device(accrucray.data, device)) * 100 \
                 for accrucray in rnn_labeled_losses.accuracies],
                time.time() - st))
            
            epoch +=1
            st = time.time()
            
if __name__ == '__main__':
    main()
