import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImage
from nnabla.ext_utils import get_extension_context
import nnabla.utils.save as save
from functools import reduce

from datasets import data_iterator_celebA
from args import get_args, save_args
from models import encoder, decoder, infer, loss_recon, loss_kl, loss_fft
from helpers import normalize_method, rgb2gray

def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)
        
    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_recon_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_kl_loss = MonitorSeries("KL Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
    monitor_image_origin = MonitorImage("Original Image", monitor, interval=1, num_images=1, 
                                        normalize_method=normalize_method)
    monitor_image_recon = MonitorImage("Reconstruction Image", monitor, interval=1, num_images=1, 
                                       normalize_method=normalize_method)

    # Solver
    solver = S.Adam(args.lr, args.beta1, args.beta2)
    
    # DataIterator
    di = data_iterator_celebA(args.train_data_path, args.batch_size)


    # Train loop
    for i in range(args.max_iter):
        # Model
        x = nn.Variable([args.batch_size, 3, args.ih, args.iw])
        e = encoder(x, args.maps)
        z, mu, logvar, var = infer(e)
        x_recon = decoder(z, args.maps * 32).apply(persistent=True)
    
        # Loss
        recon_loss = loss_recon(x_recon, x).apply(persistent=True)
        kl_loss = loss_kl(mu, logvar, var).apply(persistent=True)
        loss = recon_loss + kl_loss
    
        # Set params to solver
        solver.set_parameters(nn.get_parameters())

        # Feed data
        x_data = di.next()[0]
        x.d = x_data

        # Zerograd, forward, backward, update
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.update()
        
        # Monitor and save
        monitor_recon_loss.add(i, recon_loss.d)
        monitor_kl_loss.add(i, recon_loss.d)
        monitor_time.add(i)
        if i % args.save_interval == 0:
            monitor_image_origin.add(i, x.d)
            monitor_image_recon.add(i, x_recon.d)
            nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))

    # Monitor and save
    monitor_recon_loss.add(i, recon_loss.d)
    monitor_kl_loss.add(i, kl_loss.d)
    monitor_time.add(i)
    monitor_image.add(i, x_recon.d)
    nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))


def main():
    args = get_args()
    save_args(args, "train")

    train(args)


if __name__ == '__main__':
    main() 
