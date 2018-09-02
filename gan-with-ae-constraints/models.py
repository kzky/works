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


def pixel_wise_feature_vector_normalization(h, eps=1e-8):
    mean = F.mean(F.pow_scalar(h, 2), axis=1, keepdims=True)
    deno = F.pow_scalar(mean + eps, 0.5)
    return F.div2(h, F.broadcast(deno, h.shape))


def convblock(x, maps, kernel=(4, 4), pad=(1, 1), stride=(2, 2), 
              test=False, name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel, pad, stride)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def encoder(x, maps=16, test=False):
    with nn.parameter_scope("encoder"):
        h = x
        h = convblock(h, maps * 1, test=test, name="convblock-1")
        h = convblock(h, maps * 2, test=test, name="convblock-2")
        h = convblock(h, maps * 4, test=test, name="convblock-3")
        h = convblock(h, maps * 8, test=test, name="convblock-4")
        h = convblock(h, maps * 16, test=test, name="convblock-5")
        h = convblock(h, maps * 32, test=test, name="convblock-6")
        h = convblock(h, maps * 32, test=test, name="convblock-7")
    return h


def discriminator(x, maps=16, test=False, shared=True):
    if shared:
        return encoder(x, maps=maps, test=test)


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
    with nn.parameter_scope("decoder"):
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


def generator(x, maps=16, test=False, shared=True):
    if shared:
        return decoder(x, maps=maps, test=test)


def rgb_to_gray(x, c0=0.2989, c1=0.5870, c2=0.1140, reshape=True):
    b, c, h, w = x.shape
    r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
    gray = c0 * r + c1 * g + c2 * b
    gray = gray.reshape((b, h, w)) if reshape else gray
    return 


def laplacian_pyramid(x, level=1):
    h = x
    ds = []
    for i in range(level):
        s = F.average_pooling(h, (2, 2))
        u = F.unpooling(s, (2, 2))
        d = h - u
        h = s
        ds += [d]
    return ds


def loss_edge(x, y, level=1):
    loss = 0
    lp_xs = laplacian_pyramid(x, level)
    lp_ys = laplacian_pyramid(y, level)
    for lp_x, lp_y in zip (lp_xs, lp_ys):
        loss += F.mean(F.squared_error(lp_x, lp_y))
    return loss


def loss_rec(x_rec, x_real):
    loss = F.mean(F.squared_error(x_rec, x_real))
    return loss


def loss_kl(mu, logvar, var):
    loss = - F.mean(1 + logvar - mu ** 2 - var) * 0.5
    return loss


def loss_gan(d_x_fake, d_x_real=None):
    """Least Square Loss"""
        if d_x_real is None:
            return F.mean(F.pow_scalar((d_x_fake - 1.0), 2.0))
        return F.mean(F.pow_scalar((d_x_real - 1.0), 2.0) + F.pow_scalar(d_x_fake, 2.))


def loss_fft(x_rec, x_real, use_patch=False):
    shape = x_real.shape + (1, )
    imag = nn.Variable.from_numpy_array(np.zeros(shape))
    x_real_real = F.reshape(x_real, shape, inplace=False)
    x_real_comp = F.concatenate(x_real_real, imag, axis=len(shape) - 1)
    x_rec_real = F.reshape(x_rec, shape, inplace=False)
    x_rec_comp = F.concatenate(x_rec_real, imag, axis=len(shape) - 1)

    # Power for whole image
    x_real_fft = F.fft(x_real_comp, signal_ndim=2, normalized=True)
    x_rec_fft = F.fft(x_rec_comp, signal_ndim=2, normalized=True)
    x_real_fft_p = F.sum(x_real_fft ** 2.0, axis=shape[-1])
    x_rec_fft_p = F.sum(x_rec_fft ** 2.0, axis=shape[-1])
    loss_image = F.mean(F.squared_error(x_rec_fft_p, x_real_fft_p))

    if not use_patch:
        return loss_image

    # Power for patches
    b, h, w, _ = shape
    s = 4
    sh, sw = h // s, w // s
    loss_patch = 0
    
    for i in range(s * s):
        ih = np.random.choice(np.arange(h - sh), replace=False)
        iw = np.random.choice(np.arange(w - sw), replace=False)
        x_real_comp_patch = x_real_comp[:, ih:ih + sh, iw:iw + sw, :]
        x_rec_comp_patch = x_real_comp[:, ih:ih + sh, iw:iw + sw, :]
        x_real_patch_fft = F.fft(x_real_comp_patch, signal_ndim=2, normalized=True)
        x_rec_patch_fft = F.fft(x_rec_comp_patch, signal_ndim=2, normalized=True)
        x_real_patch_fft_p = F.sum(x_real_patch_fft ** 2.0, axis=shape[-1])
        x_rec_patch_fft_p = F.sum(x_rec_patch_fft ** 2.0, axis=shape[-1])
        l = F.mean(F.squared_error(x_rec_patch_fft_p, x_real_patch_fft_p))
        loss_patch += l
    return loss_image + loss_patch


def main():
    # Data
    b, c, h, w = 8, 3, 128, 128
    maps = 16
    x = nn.Variable([b, c, h, w])
    # Network
    e = encoder(x, maps)
    print("Encode:", e)
    z, mu, logvar, var = infer(e)
    x_rec = decoder(z, maps * 32)
    print("Rec", x_rec)
    # Loss
    rec_loss = loss_rec(x_rec, x)
    kl_loss = loss_kl(mu, logvar, var)
    loss = rec_loss + kl_loss

    print("Loss", loss)

if __name__ == '__main__':
    main()


