import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla
from nnabla.contrib.context import extension_context
import numpy as np
import os
import time
import argparse
from st2.cifar10.cnn_model_023 import cnn_model_003, ce_loss, sr_loss
from st2.cifar10.datasets import Cifar10DataReader, Separator

"""
The same script as the `st` module but with nnabla.

- ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
- Stochastic Regularization
"""

def batch_stochastic_supervised_network(ctx, batch_sizes, c, h, w):
    x_list = []
    y_list = []
    preds = []
    losses = []
    for b in batch_sizes:
        x = nn.Variable((b, c, h, w))
        y = nn.Variable((b, 1))
        pred = cnn_model_003(ctx, x)
        loss_ce = ce_loss(ctx, pred, y)
        
        x_list.append(x)
        y_list.append(y)
        preds.append(pred)
        losses.append(loss_ce)
    return x_list, y_list, preds, losses

def batch_stochastic_unsupervised_network(ctx, batch_sizes, c, h, w):
    x0_list = []
    x1_list = []
    losses = []
    for b in batch_sizes:
        x0 = nn.Variable((b, c, h, w))
        x1 = nn.Variable((b, c, h, w))
        pred_x0 = cnn_model_003(ctx, x0)
        pred_x1 = cnn_model_003(ctx, x1)
        loss_sr = sr_loss(ctx, pred_x0, pred_x1)
        x0_list.append(x0)
        x1_list.append(x1)
        losses.append(loss_sr)
    return x0_list, x1_list, None, losses

def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()

def main(args):
    # Settings
    device_id = args.device_id
    batch_sizes = [16, 32, 64]
    batch_size_eval = 64
    c, h, w = 3, 32, 32
    n_l_train_data = 4000
    n_train_data = 50000
    n_cls = 10
    learning_rate = 1. * 1e-3
    n_epoch = 300
    act = F.relu
    iter_epoch = n_train_data / int(np.mean(batch_sizes))  # approximate epoch
    n_iter = n_epoch * iter_epoch
    extension_module = args.context

    # Model (Batch-Stochastic)
    ctx = extension_context(extension_module, device_id=device_id)
    ## supervised
    x_list, y_list, preds, losses = batch_stochastic_supervised_network(
        ctx, batch_sizes, c, h, w)
    
    ## stochastic regularization
    x0_list, x1_list, _, losses = batch_stochastic_unsupervised_network(
        ctx, batch_sizes, c, h, w)

    ## evaluate
    batch_size_eval, m, h, w = batch_size_eval, c, h, w
    x_eval = nn.Variable((batch_size_eval, m, h, w))
    pred_eval = cnn_model_003(ctx, x_eval, test=True)
    
    # Solver
    with nn.context_scope(ctx):
        solver = S.Adam(alpha=learning_rate)
        solver.set_parameters(nn.get_parameters())

    # Dataset
    ## separate dataset
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/cifar10/cifar-10.npz")
    separator = Separator(n_l_train_data)
    separator.separate_then_save(fpath)

    l_train_path = os.path.join(home, "datasets/cifar10/l_cifar-10.npz")
    u_train_path = os.path.join(home, "datasets/cifar10/cifar-10.npz")
    test_path = os.path.join(home, "datasets/cifar10/cifar-10.npz")

    # data reader
    data_reader = Cifar10DataReader(l_train_path, u_train_path, test_path,
                                  batch_size=batch_sizes[0],
                                  n_cls=n_cls,
                                  da=True,
                                  shape=True)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    acc_prev = 0.
    iter_ = 0
    for i in range(n_iter):
        idx = np.random.choice(np.arange(0, len(batch_sizes)))
        # Get data
        batch_size = batch_sizes[idx]
        x_l0_data, x_l1_data, y_l_data = data_reader.get_l_train_batch(batch_size)
        x_u0_data, x_u1_data, y_u_data = data_reader.get_u_train_batch(batch_size)

        #  Set it to the varaibles
        x_l = x_list[idx]
        y_l = y_list[idx]
        x_u0 = x0_list[idx]
        x_u1 = x1_list[idx]
        x_l.d, _ , y_l.d= x_l0_data, x_l1_data, y_l_data
        x_u0.d, x_u1.d= x_u0_data, x_u1_data

        # Train
        loss_ce.forward(clear_no_need_grad=True)
        loss_sr.forward(clear_no_need_grad=True)
        solver.zero_grad()
        loss_ce.backward(clear_buffer=True)
        loss_sr.backward(clear_buffer=True)
        solver.update()
        
        # Evaluate
        if (i+1) % iter_epoch == 0:  # approximate epoch
            # Get data and set it to the varaibles
            x_data, y_data = data_reader.get_test_batch()

            # Evaluation loop
            ve = 0.
            iter_val = 0
            for k in range(0, len(x_data), batch_size_eval):
                x_eval.d = x_data[k:k+batch_size_eval, :]
                label = y_data[k:k+batch_size_eval, :]
                pred_eval.forward(clear_buffer=True)
                ve += categorical_error(pred_eval.d, label)
                iter_val += 1
            msg = "Epoch:{},ElapsedTime:{},Acc:{:02f}".format(
                epoch,
                time.time() - st, 
                (1. - ve / iter_val) * 100)
            print(msg)
            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", "-d", type=int, default=0)
    parser.add_argument('--context', '-c', type=str,
                        default="cpu", help="Extension modules. ex) 'cpu', 'cuda.cudnn'.")
    args = parser.parse_args()

    main(args)
