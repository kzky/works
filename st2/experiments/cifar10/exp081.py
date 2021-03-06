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
from st2.cifar10.cnn_model_081 import cnn_model_003, ce_loss, sr_loss, er_loss, sr_loss_with_uncertainty, ce_loss_with_uncertainty, sigma_regularization, sigmas_regularization, cifar10_resnet23_prediction, kl_divergence
from st2.cifar10.datasets import Cifar10DataReader, Separator

"""
The same script as the `st` module but with nnabla.

- ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
- Based on results of exp051.py
- Transfer knowledge to ResNet23 w/ temperature
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
    ## supervised resnet
    batch_size, m, h, w = batch_size, 3, 32, 32
    ctx = extension_context(extension_module, device_id=device_id)
    x_l = nn.Variable((batch_size, m, h, w))
    y_l = nn.Variable((batch_size, 1))
    pred_res = cifar10_resnet23_prediction(ctx, "resnet", x_l)
    loss_res_ce = ce_loss(ctx, pred_res, y_l)
    loss_res_supervised = loss_res_ce

    ## stochastic regularization
    nn.load_parameters(args.model_load_path)
    x_u0.persistent = True
    x_u1 = nn.Variable((batch_size, m, h, w))
    pred_x_u0, log_var0 = cnn_model_003(ctx, x_u0)
    pred_x_u0.need_grad, log_var0.need_grad = False, False

    ## knowledge transfer for resnet
    pred_res_x_u0 = cifar10_resnet23_prediction(ctx, "resnet", x_u0)
    loss_res_unsupervised = kl_divergence(ctx, pred_res_x_u0, pred_x_u0, log_var0)

    ## evaluate
    batch_size_eval, m, h, w = batch_size, 3, 32, 32
    x_eval = nn.Variable((batch_size_eval, m, h, w))
    pred_res_eval = cifar10_resnet23_prediction(ctx, "resnet", x_eval, test=True)

    # Solver
    with nn.context_scope(ctx):
        with nn.parameter_scope("resnet"):
            solver_res = S.Adam(alpha=learning_rate)
            solver_res.set_parameters(nn.get_parameters())

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

        # Train resnet
        loss_res_supervised.forward(clear_no_need_grad=True)
        loss_res_unsupervised.forward(clear_no_need_grad=True)
        solver_res.zero_grad()
        loss_res_supervised.backward(clear_buffer=True)
        loss_res_unsupervised.backward(clear_buffer=True)
        solver_res.update()

        # Evaluate
        if (i+1) % iter_epoch == 0:
            # Get data and set it to the varaibles
            x_data, y_data = data_reader.get_test_batch()

            # Evaluation loop for resnet
            ve = 0.
            iter_val = 0
            for k in range(0, len(x_data), batch_size_eval):
                x_eval.d = get_test_data(x_data, k, batch_size_eval)
                label = get_test_data(y_data, k, batch_size_eval)
                pred_res_eval.forward(clear_buffer=True)
                ve += categorical_error(pred_res_eval.d, label)
                iter_val += 1
            msg = "Model:resnet,Epoch:{},ElapsedTime:{},Acc:{:02f}".format(
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
    parser.add_argument("--model-load-path", "-l", type=str, default="")
    args = parser.parse_args()

    main(args)
