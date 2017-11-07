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
from st2.cifar10.cnn_model_005_001 import cnn_model_003, ce_loss, sr_loss, er_loss, ce_loss_soft
from st2.cifar10.datasets import Cifar10DataReader, Separator

"""
The same script as the `st` module but with nnabla.

- ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
- Stochastic Regularization
- Entropy Regularization for the outputs before CE loss and SR loss
- Mixup (augment data, then mixup)
- Entropy Regularization for the outputs of mixup prediction
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
    iter_epoch = int(n_train_data / batch_size)
    n_iter = n_epoch * iter_epoch
    extension_module = args.context
    alpha = args.alpha

    # Supervised Model 
    ## ERM
    batch_size, m, h, w = batch_size, 3, 32, 32
    ctx = extension_context(extension_module, device_id=device_id)
    x_l_0 = nn.Variable((batch_size, m, h, w))
    y_l_0 = nn.Variable((batch_size, 1))
    pred = cnn_model_003(ctx, x_l_0)
    loss_ce = ce_loss(ctx, pred, y_l_0)
    loss_er = er_loss(ctx, pred)
    loss_supervised = loss_ce + loss_er
    ## VRM (mixup)
    x_l_1 = nn.Variable((batch_size, m, h, w))
    y_l_1 = nn.Variable((batch_size, 1))
    coef = nn.Variable()
    coef_b = F.broadcast(coef.reshape([1]*x_l_0.ndim, unlink=True), x_l_0.shape)
    x_l_m = coef_b * x_l_0 + (1 - coef_b) * x_l_1
    coef_b = F.broadcast(coef.reshape([1]*pred.ndim, unlink=True), pred.shape)
    y_l_m = coef_b * F.one_hot(y_l_0, (n_cls, )) \
            + (1-coef_b) * F.one_hot(y_l_1, (n_cls, ))
    x_l_m.need_grad, y_l_m.need_grad = False, False
    pred_m = cnn_model_003(ctx, x_l_m)
    loss_er_m = er_loss(ctx, pred_m)  #todo: need?
    loss_ce_m = ce_loss_soft(ctx, pred, y_l_m)
    loss_supervised_m = loss_ce_m #+ loss_er_m
    
    # Semi-Supervised Model
    ## ERM
    x_u0 = nn.Variable((batch_size, m, h, w))
    x_u1 = nn.Variable((batch_size, m, h, w))
    pred_x_u0 = cnn_model_003(ctx, x_u0)
    pred_x_u1 = cnn_model_003(ctx, x_u1)
    pred_x_u0.persistent, pred_x_u1.persistent = True, True
    loss_sr = sr_loss(ctx, pred_x_u0, pred_x_u1)
    loss_er0 = er_loss(ctx, pred_x_u0)
    loss_er1 = er_loss(ctx, pred_x_u1)
    loss_unsupervised = loss_sr + loss_er0 + loss_er1
    ## VRM (mixup)
    x_u2 = nn.Variable((batch_size, m, h, w))  # not to overwrite x_u1.d
    coef_u = nn.Variable()
    coef_u_b = F.broadcast(coef_u.reshape([1]*x_u0.ndim, unlink=True), x_u0.shape)
    x_u_m = coef_u_b * x_u0 + (1-coef_u_b) * x_u2
    pred_x_u0_ = nn.Variable(pred_x_u0.shape)  # unlink forward pass but reuse result
    pred_x_u1_ = nn.Variable(pred_x_u1.shape)
    pred_x_u0_.data = pred_x_u0.data
    pred_x_u1_.data = pred_x_u1.data
    coef_u_b = F.broadcast(coef_u.reshape([1]*pred_x_u0.ndim, unlink=True), pred_x_u0.shape)
    y_u_m = coef_u_b * pred_x_u0_ + (1-coef_u_b) * pred_x_u1_
    x_u_m.need_grad, y_u_m.need_grad = False, False
    pred_x_u_m = cnn_model_003(ctx, x_u_m)
    loss_er_u_m = er_loss(ctx, pred_x_u_m)  #todo: need?
    loss_ce_u_m = ce_loss_soft(ctx, pred_x_u_m, y_u_m)
    loss_unsupervised_m = loss_ce_u_m #+ loss_er_u_m
    
    # Evaluatation Model
    batch_size_eval, m, h, w = batch_size, 3, 32, 32
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
                                  batch_size=batch_size,
                                  n_cls=n_cls,
                                  da=True,
                                  shape=True)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    acc_prev = 0.
    ve_best = 1.
    save_path_prev = ""
    for i in range(n_iter):
        # Get data and set it to the varaibles
        x_l0_data, x_l1_data, y_l_data = data_reader.get_l_train_batch()
        x_u0_data, x_u1_data, y_u_data = data_reader.get_u_train_batch()
        
        x_l_0.d, _ , y_l_0.d= x_l0_data, x_l1_data, y_l_data
        x_u0.d, x_u1.d= x_u0_data, x_u1_data

        # Train
        ## forward (supervised and its mixup)
        loss_supervised.forward(clear_no_need_grad=True)
        coef_data = np.random.beta(alpha, alpha)
        coef.d = coef_data
        x_l_1.d = np.random.permutation(x_l0_data)
        y_l_1.d = np.random.permutation(y_l_data)
        loss_supervised_m.forward(clear_no_need_grad=True)
        ## forward (unsupervised and its mixup)
        loss_unsupervised.forward(clear_no_need_grad=True)
        coef_data = np.random.beta(alpha, alpha)
        coef_u.d = coef_data
        x_u2.d = np.random.permutation(x_u1_data)
        loss_unsupervised_m.forward(clear_no_need_grad=True)
        
        ## backward
        solver.zero_grad()
        loss_supervised.backward(clear_buffer=False)
        loss_supervised_m.backward(clear_buffer=False)
        loss_unsupervised.backward(clear_buffer=False)
        loss_unsupervised_m.backward(clear_buffer=True)
        solver.update()
        
        # Evaluate
        if int((i+1) % iter_epoch) == 0:
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
            ve /= iter_val                
            msg = "Epoch:{},ElapsedTime:{},Acc:{:02f}".format(
                epoch,
                time.time() - st, 
                (1. - ve) * 100)
            print(msg)
            if ve < ve_best:
                if not os.path.exists(args.model_save_path):
                    os.makedirs(args.model_save_path)
                if save_path_prev != "":
                    os.remove(save_path_prev)
                save_path = os.path.join(
                    args.model_save_path, 'params_%06d.h5' % epoch)
                nn.save_parameters(save_path)
                save_path_prev = save_path
                ve_best = ve
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
    parser.add_argument("--model-save-path", "-o",
                        type=str, default="tmp.monitor",
                        help='Path where model parameters are saved.')
    parser.add_argument("--alpha", "-a", type=int, default=0.2)
    args = parser.parse_args()

    main(args)
