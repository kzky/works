import os
import functools
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.parameter import get_parameter_or_create
from nnabla.ext_utils import get_extension_context
from nnabla.parametric_functions import parametric_function_api

import nnabla.initializer as I

#TODO: conditioning

def convblock(x, maps, kernel=(4, 4), pad=(1, 1), stride=(2, 2), 
              test=False, name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel, pad, stride)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def encoder(x, maps=16, test=False):
    h = x
    h = convblock(h, maps * 1, test=test, name="convblock-1")
    h = convblock(h, maps * 2, test=test, name="convblock-2")
    h = convblock(h, maps * 4, test=test, name="convblock-3")
    h = convblock(h, maps * 8, test=test, name="convblock-4")
    h = convblock(h, maps * 16, test=test, name="convblock-5")
    h = convblock(h, maps * 32, test=test, name="convblock-6")
    h = convblock(h, maps * 32, test=test, name="convblock-7")
    return h


def deconvblock(x, maps, kernel=(4, 4), pad=(1, 1), stride=(2, 2), use_deconv=False, 
                test=False, name="convblock"):
    h = x
    with nn.parameter_scope(name):
        if use_deconv:
            h = PF.deconvolution(h, maps, kernel, pad, stride)
        else:
            h = F.unpooling(h, (2, 2))
            h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def decoder(z, maps=512, test=False):
    h = z
    h = deconvblock(h, maps // 1, test=test, name="deconvblock-1")
    h = deconvblock(h, maps // 1, test=test, name="deconvblock-2")
    h = deconvblock(h, maps // 2, test=test, name="deconvblock-3")
    h = deconvblock(h, maps // 4, test=test, name="deconvblock-4")
    h = deconvblock(h, maps // 8, test=test, name="deconvblock-5")
    h = deconvblock(h, maps // 16, test=test, name="deconvblock-6")
    h = deconvblock(h, maps // 32, test=test, name="deconvblock-7")
    h = PF.convolution(h, 3, kernel=(3, 3), pad=(1, 1), name="last-conv")
    return h


def infer(x, sigma=1.0, T=100):
    #TODO: sphere-constraint
    b, c, h, w = x.shape
    mu = PF.convolution(x, c, kernel=(1, 1), pad=(0, 0))
    logvar = PF.convolution(x, c, kernel=(1, 1), pad=(0, 0))
    #TODO: better to clip?
    #logvar = F.minimum_scalar(F.maximum_scalar(logvar, -T), T)
    var = F.exp(logvar)
    std = F.pow_scalar(var, 0.5)
    n = F.randn(sigma=sigma, shape=(b, c, h, w))
    z = x + std * n
    return z, mu, logvar, var


def loss_recon(x_recon, x_real):
    loss = F.mean(F.squared_error(x_recon, x_real))
    return loss


def loss_kl(mu, logvar, var):
    loss = - F.mean(1 + logvar - mu ** 2 - var) * 0.5
    return loss


def main():
    # Data
    b, c, h, w = 8, 3, 128, 128
    maps = 16
    x = nn.Variable([b, c, h, w])
    # Network
    e = encoder(x, maps)
    print("Encode:", e)
    z, mu, logvar, var = infer(e)
    x_recon = decoder(z, maps * 32)
    print("Recon", x_recon)
    # Loss
    recon_loss = loss_recon(x_recon, x)
    kl_loss = loss_kl(mu, logvar, var)
    loss = recon_loss + kl_loss

    print("Loss", loss)

if __name__ == '__main__':
    main()


