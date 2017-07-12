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
from st2.cifar10.cnn_model_015 import cnn_model_003, ce_loss, sr_loss, er_loss, \
    GradScaleContainer
from st2.cifar10.datasets import Cifar10DataReader, Separator

"""
The same script as the `st` module but with nnabla.

- ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
- Stochastic Regularization
- Entropy Regularization for the outputs before CE loss and SR loss
- Naive implementation for gradient scaling; basis is grad scale of supervised loss
- Solver is separated
- batch stats are different among the three networks and use the last in inference.
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
    act = F.relu
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch
    extension_module = args.context

    # Model
    cnt = 0
    ## supervised 
    batch_size, m, h, w = batch_size, 3, 32, 32
    ctx = extension_context(extension_module, device_id=device_id)
    x_l = nn.Variable((batch_size, m, h, w))
    y_l = nn.Variable((batch_size, 1))
    pred = cnn_model_003(ctx, x_l, cnt++)
    loss_ce = ce_loss(ctx, pred, y_l)
    loss_er = er_loss(ctx, pred)
    loss_supervised = loss_ce + loss_er

    ## stochastic regularization
    x_u0 = nn.Variable((batch_size, m, h, w))
    x_u1 = nn.Variable((batch_size, m, h, w))
    pred_x_u0 = cnn_model_003(ctx, x_u0, cnt++)
    pred_x_u1 = cnn_model_003(ctx, x_u1, cnt++)
    loss_sr = sr_loss(ctx, pred_x_u0, pred_x_u1)
    loss_er0 = er_loss(ctx, pred_x_u0)
    loss_er1 = er_loss(ctx, pred_x_u1)
    loss_unsupervised = loss_sr + loss_er0 + loss_er1

    ## evaluate
    batch_size_eval, m, h, w = batch_size, 3, 32, 32
    x_eval = nn.Variable((batch_size_eval, m, h, w))
    pred_eval = cnn_model_003(ctx, x_eval, test=True, cnt)
    
    # Solver
    with nn.context_scope(ctx):
        solver_superrvised = S.Adam(alpha=learning_rate)
        solver_superrvised.set_parameters(nn.get_parameters())
        solver_unsuperrvised = S.Adam(alpha=learning_rate)
        solver_unsuperrvised.set_parameters(nn.get_parameters())

    # Gradient Scale Container
    gsc = GradScaleContainer(len(nn.get_parameters()))

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
                                  da=True,  #TODO: use F.image_augmentation
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
        ## compute scales and update with grads of supervised loss
        solver_superrvised.zero_grad()
        loss_supervised.backward(clear_buffer=True)
        gsc.set_scales_supervised_loss(nn.get_parameters())
        solver_superrvised.update()
        ## compute scales and update with grads of unsupervised loss
        solver_unsuperrvised.zero_grad()
        loss_unsupervised.backward(clear_buffer=True)
        gsc.set_scales_unsupervised_loss(nn.get_parameters())
        gsc.scale_grad(ctx, nn.get_parameters())
        solver_unsuperrvised.update()
        
        # Evaluate
        if (i+1) % iter_epoch == 0:
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
