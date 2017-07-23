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
from st2.cifar10.cnn_model_018 import resnet_model, ce_loss, sr_loss
from st2.cifar10.datasets import Cifar10DataReader, Separator

"""
The same script as the `st` module but with nnabla.

- ResNet
- Stochastic Regularization
"""

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
    batch_size = 100
    batch_size_eval = 100
    n_l_train_data = 4000
    n_train_data = 50000
    n_cls = 10
    learning_rate = 1. * 1e-3
    n_epoch = 300
    inmaps = 128
    act = F.relu
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch
    extension_module = args.context

    # Model
    ## supervised 
    batch_size, m, h, w = batch_size, 3, 32, 32
    ctx = extension_context(extension_module, device_id=device_id)
    x_l = nn.Variable((batch_size, m, h, w))
    y_l = nn.Variable((batch_size, 1))
    pred = resnet_model(ctx, x_l, inmaps, act)
    loss_ce = ce_loss(ctx, pred, y_l)

    ## stochastic regularization
    x_u0 = nn.Variable((batch_size, m, h, w))
    x_u1 = nn.Variable((batch_size, m, h, w))
    pred_x_u0 = resnet_model(ctx, x_u0, inmaps, act)
    pred_x_u1 = resnet_model(ctx, x_u1, inmaps, act)
    loss_sr = sr_loss(ctx, pred_x_u0, pred_x_u1)

    ## evaluate
    batch_size_eval, m, h, w = batch_size, 3, 32, 32
    x_eval = nn.Variable((batch_size_eval, m, h, w))
    pred_eval = resnet_model(ctx, x_eval, inmaps, act, test=True)
    
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
                                  batch_size=batch_size,
                                  n_cls=n_cls,
                                  da=True,
                                  shape=True)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    acc_prev = 0.
    for i in range(n_iter):
        # Get data and set it to the varaibles
        x_l0_data, x_l1_data, y_l_data = data_reader.get_l_train_batch()
        x_u0_data, x_u1_data, y_u_data = data_reader.get_u_train_batch()
        
        x_l.d, _ , y_l.d= x_l0_data, x_l1_data, y_l_data
        x_u0.d, x_u1.d= x_u0_data, x_u1_data

        # Train
        with nn.auto_forward():
            # for CE
            pred = resnet_model(ctx, x_l, inmaps, act)
            loss_ce = ce_loss(ctx, pred, y_l)

            # for SR
            pred_x_u0 = resnet_model(ctx, x_u0, inmaps, act)
            pred_x_u1 = resnet_model(ctx, x_u1, inmaps, act)
            loss_sr = sr_loss(ctx, pred_x_u0, pred_x_u1)
        
        solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
        solver.zero_grad()
        loss_ce.backward(clear_buffer=True)
        loss_sr.backward(clear_buffer=True)
        solver.update()
        
        # Evaluate
        if (i+1) % iter_epoch == 0:
            # Get data and set it to the varaibles
            x_data, y_data = data_reader.get_test_batch()

            # Evaluation loop
            ve = 0.
            iter_val = 0
            for k in range(0, len(x_data), batch_size_eval):
                x_eval.d = get_test_data(x_data, k, batch_size_eval)
                label = get_test_data(y_data, k, batch_size_eval)
                pred_eval = resnet_model(ctx, x_eval, inmaps, act, test=True)

                ve += categorical_error(pred_eval.d, label)
                iter_val += 1
            msg = "Epoch:{},ElapsedTime:{},Acc:{:02f}".format(
                epoch,
                time.time() - st, 
                (1. - ve / iter_val) * 100)
            print(msg)
            st = time.time()
            epoch +=1

def get_test_data(data, k, batch_size):
    data_ = data[k:k+batch_size, :]
    if len(data_) == batch_size:
        return data_
    data_ = np.concatenate(
        (data[0:batch_size-len(data_), :], data_)
    )
    return data_
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", "-d", type=int, default=0)
    parser.add_argument('--context', '-c', type=str,
                        default="cpu", help="Extension modules. ex) 'cpu', 'cuda.cudnn'.")
    args = parser.parse_args()

    main(args)
