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
from st2.cifar10.cnn_model_050 import cnn_model_003, ce_loss, sr_loss, er_loss, sr_loss_with_uncertainty, ce_loss_with_uncertainty, sigma_regularization, sigma_sigma_regularization
from st2.cifar10.datasets import Cifar10DataReader, Separator

"""
The same script as the `st` module but with nnabla.

- ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
- Stochastic Regularization
- Enstropy Regularization
- Uncertainty for SR loss using JSD
- Uncertainty for CE
- Squared error between uncertainty (sigma) and one, sigma becmoes one at the end. Applied for CE loss also.
- Add coefficient to sigma regularization
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
    batch_size = args.batch_size
    batch_size_eval = args.batch_size_eval
    n_l_train_data = 4000
    n_train_data = 50000
    n_cls = 10
    learning_rate = 1. * 1e-3
    n_epoch = 300
    act = F.relu
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch
    extension_module = args.context
    lambda_ = args.lambda_

    # Model
    ## supervised 
    batch_size, m, h, w = batch_size, 3, 32, 32
    ctx = extension_context(extension_module, device_id=device_id)
    x_l = nn.Variable((batch_size, m, h, w))
    y_l = nn.Variable((batch_size, 1))
    pred, log_var, log_s = cnn_model_003(ctx, x_l)
    one = F.constant(1., log_var.shape)
    loss_ce = ce_loss_with_uncertainty(ctx, pred, y_l, log_var)
    reg_sigma = sigma_regularization(ctx, log_var, one)
    loss_supervised = loss_ce + er_loss(ctx, pred) + lambda_ * reg_sigma

    ## stochastic regularization
    x_u0 = nn.Variable((batch_size, m, h, w))
    x_u1 = nn.Variable((batch_size, m, h, w))
    pred_x_u0, log_var0, log_s0 = cnn_model_003(ctx, x_u0)
    pred_x_u1, log_var1, log_s1 = cnn_model_003(ctx, x_u1)
    zero = F.constant(0., log_s0.shape)
    loss_sr = sr_loss_with_uncertainty(ctx, 
                                       pred_x_u0, pred_x_u1, log_var0, log_var1, log_s0, log_s1)
    reg_sigma0 = sigma_regularization(ctx, log_var0, one)
    reg_sigma1 = sigma_regularization(ctx, log_var1, one)
    reg_sigma_sigma0 = sigma_regularization(ctx, log_s0, zero)
    reg_sigma_sigma1 = sigma_regularization(ctx, log_s1, zero)
    loss_unsupervised = loss_sr + er_loss(ctx, pred_x_u0) + er_loss(ctx, pred_x_u1) \
                        + lambda_ * (reg_sigma0 + reg_sigma1) \
                        + lambda_ * (reg_sigma_sigma0 + reg_sigma_sigma1)
    ## evaluate
    batch_size_eval, m, h, w = batch_size, 3, 32, 32
    x_eval = nn.Variable((batch_size_eval, m, h, w))
    pred_eval, _ = cnn_model_003(ctx, x_eval, test=True)
    
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
        loss_supervised.forward(clear_no_need_grad=True)
        loss_unsupervised.forward(clear_no_need_grad=True)
        solver.zero_grad()
        loss_supervised.backward(clear_buffer=True)
        loss_unsupervised.backward(clear_buffer=True)
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
    parser.add_argument("--batch_size", "-b", type=int, default=100)
    parser.add_argument("--batch_size_eval", "-e", type=int, default=100)
    parser.add_argument("--lambda_", "-l", type=float, default=1.)
    args = parser.parse_args()

    main(args)
