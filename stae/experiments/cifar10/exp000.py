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
from stae.cifar10.cnn_ae_model_000 import cnn_ae_model_000, recon_loss
from stae.cifar10.datasets import Cifar10DataReader, Separator
import cv2

"""
- Stochastic Autoencoder (using dropout)
"""

def save_images(dpath, epoch, images):
    images *= 255.
    for i, image in enumerate(images):
        fpath = os.path.join(
            dpath, "epoch_{:05d}-index_{:05d}.png".format(epoch, i))
        image = cv2.cvtColor(image.transpose((1, 2, 0)), 
                             cv2.COLOR_RGB2BGR)
        cv2.imwrite(fpath, image)

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
    n_images = args.n_images 
    fname, _ = os.path.splitext(__file__)
    dpath = "./{}_images_{}".format(fname, int(time.time()))

    # Model
    batch_size, m, h, w = batch_size, 3, 32, 32
    ctx = extension_context(extension_module, device_id=device_id)
    x_u = nn.Variable((batch_size, m, h, w))
    pred = cnn_ae_model_000(ctx, x_u)
    loss_recon = recon_loss(ctx, pred, x_u)

    ## evaluate
    batch_size_eval, m, h, w = batch_size, 3, 32, 32
    x_eval = nn.Variable((batch_size_eval, m, h, w))
    pred_eval = cnn_ae_model_000(ctx, x_eval, test=True)
    
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
        x_u_data, _, _ = data_reader.get_u_train_batch()
        x_u.d = x_u_data

        # Train
        loss_recon.forward(clear_no_need_grad=True)
        solver.zero_grad()
        loss_recon.backward(clear_buffer=True)
        solver.update()
        
        # Evaluate
        if (i+1) % iter_epoch == 0:
            # Get data and forward
            x_data, y_data = data_reader.get_test_batch()
            pred.forward(clear_buffer=True)
            images = pred.d

            # Save n images
            if not os.path.exists(dpath):
                os.makedirs(dpath)
            save_images(dpath, epoch, images[:n_images])
            fpath = os.path.join(dpath, "epoch_{:05d}.h5".format(epoch))
            nn.save_parameters(fpath)

            st = time.time()
            epoch +=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", "-d", type=int, default=0)
    parser.add_argument('--context', '-c', type=str,
                        default="cpu", help="Extension modules. ex) 'cpu', 'cuda.cudnn'.")
    parser.add_argument('--num-images', '-n', type=int, dest="n_images", 
                        default=10)
    args = parser.parse_args()

    main(args)
