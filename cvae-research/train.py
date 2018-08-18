import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImag
from nnabla.ext_utils import get_extension_context
import nnabla.utils.save as save
from functools import reduce

from datasets import data_iterator_celebA
from args import get_args, save_args
from models import get_loss, lapsrn
from helpers import (get_solver, upsample, downsample, 
                     split, to_BCHW, to_BHWC, normalize, ycbcr_to_rgb, 
                     normalize_method)


def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    x = nn.Variable([b, c, h, w])
    e = encoder(x, maps)
    z, mu, logvar, var = infer(e)
    x_recon = decoder(z, c * 32).apply(persistent=True)

    # Loss
    recon_loss = loss_recon(x_recon, x).apply(persistent=True)
    kl_loss = loss_kl(mu, logvar, var).apply(persistent=True)
    loss = recon_loss + kl_loss

    # Solver
    solver = S.Adam(args.lr, args.beta1, args.beta2)
    solver.set_parameters(nn.get_parameters())
        
    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_recon_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_kl_loss = MonitorSeries("KL Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
    monitor_image = MonitorImage("Reconstruction Image", monitor, interval=1)

    # DataIterator
    di = data_iterator_celebA(args.img_path, args.batch_size)
    
    # Train loop
    for i in range(args.max_iter):
        # Feed data
        x_data = di.next()[0]

        # Zerograd, forward, backward, weight-decay, update
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.update()
        
        # Monitor and save
        monitor_recon_loss.add(i, recon_loss.d)
        monitor_kl_loss.add(i, recon_loss.d)
        monitor_time.add(i)
        if i % args.save_interval == 0:
            monitor_image.add(i, x_recon.d)

    # Monitor and save
    monitor_recon_loss.add(i, recon_loss.d)
    monitor_kl_loss.add(i, recon_loss.d)
    monitor_time.add(i)
    if i % args.save_interval == 0:
        monitor_image.add(i, x_recon.d)


def main():
    args = get_args()
    save_args(args, "train")

    train(args)


if __name__ == '__main__':
    main() 
