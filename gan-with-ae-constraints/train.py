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
from models import (encoder, decoder, generator, discriminator, 
                    loss_rec, loss_edge, loss_gan, 
                    pixel_wise_feature_vector_normalization)
from helpers import normalize_method, rgb2gray

def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)
        
    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_rec_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_gen_loss = MonitorSeries("Generator Loss", monitor, interval=10)
    monitor_dis_loss = MonitorSeries("Discriminator Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
    monitor_image_origin = MonitorImage("Original Image", monitor, interval=1, num_images=1, 
                                        normalize_method=normalize_method)
    monitor_image_rec = MonitorImage("Reconstruction Image", monitor, interval=1, num_images=1, 
                                       normalize_method=normalize_method)
    monitor_image_gen = MonitorImage("Generate Image", monitor, interval=1, num_images=1, 
                                     normalize_method=normalize_method)
        
    # Model
    x_real0 = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    x_real1 = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    e, h_list = encoder(x_real0, args.maps)
    e = pixel_wise_feature_vector_normalization(e) if args.use_pfvn else e
    x_rec = decoder(e, args.maps * 32, h_list=h_list).apply(persistent=True)
    r = F.randn(shape=e.shape)
    z = e + pixel_wise_feature_vector_normalization(r) if args.use_pfvn else r
    z = pixel_wise_feature_vector_normalization(z).apply(need_grad=False) \
        if args.use_pfvn else z.apply(need_grad=False)
    x_fake = generator(z, test=False, h_list=h_list)
    d_fake, _ = discriminator(x_fake, test=False)
    d_real, _ = discriminator(x_real1, test=False)
    
    # Loss
    vloss_rec = loss_rec(x_rec, x_real0).apply(persistent=True)
    vloss_gen = args.lam * loss_gan(d_fake).apply(persistent=True)
    vloss_dis = args.lam * loss_gan(d_fake, d_real).apply(persistent=True)
        
    # Solver
    solver_enc = S.Adam(args.lr, args.beta1, args.beta2)
    with nn.parameter_scope("encoder"):
        solver_enc.set_parameters(nn.get_parameters())
    solver_dec = S.Adam(args.lr, args.beta1, args.beta2)
    with nn.parameter_scope("decoder"):
        solver_dec.set_parameters(nn.get_parameters())
    solver_gen = solver_dec
    solver_dis = solver_enc

    # DataIterator
    di = data_iterator_celebA(args.train_data_path, args.batch_size)

    # Train loop
    for i in range(args.max_iter):
        # Feed data
        x_data = di.next()[0]
        x_real0.d = x_data
        # Train Auto-Encoder
        solver_enc.zero_grad()
        solver_dec.zero_grad()
        vloss_rec.forward(clear_no_need_grad=True)
        vloss_rec.backward(clear_buffer=True)
        solver_enc.update()
        solver_dec.update()
        # Train Generator
        solver_gen.zero_grad()
        vloss_gen.forward(clear_no_need_grad=True)
        vloss_gen.backward(clear_buffer=True)
        solver_gen.update()

        # Feed data
        x_data = di.next()[0]
        x_real1.d = x_data
        # Train Discriminator
        solver_dis.zero_grad()
        vloss_dis.forward(clear_no_need_grad=True)
        vloss_dis.backward(clear_buffer=True)
        solver_dis.update()
        
        # Monitor and save
        monitor_rec_loss.add(i, vloss_rec.d)
        monitor_gen_loss.add(i, vloss_gen.d)
        monitor_dis_loss.add(i, vloss_dis.d)
        monitor_time.add(i)
        if i % args.save_interval == 0:
            monitor_image_origin.add(i, x_real0.d)
            monitor_image_rec.add(i, x_rec.d)
            monitor_image_gen.add(i, x_fake.d)
            nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))

    # Monitor and save
    monitor_rec_loss.add(i, rec_loss.d)
    monitor_time.add(i)
    monitor_image.add(i, x_rec.d)
    nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))


def main():
    args = get_args()
    save_args(args, "train")

    train(args)


if __name__ == '__main__':
    main() 
