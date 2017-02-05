"""Models
"""
import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from utils import to_device
from chainer_fix import BatchNormalization

class Generator(Chain):
    """Residual Learning as Super Resolution.
    """
    def __init__(self, device=None, act=F.relu, n_cls=10, dims=100):
        super(Generator, self).__init__(
            # Encoder
            conv0=L.Convolution2D(1+n_cls, 32, ksize=4, stride=2, pad=1),
            conv1=L.Convolution2D(32, 64, ksize=4, stride=2, pad=1),
            bn_conv0=L.BatchNormalization(32, decay=0.9),
            bn_conv1=L.BatchNormalization(64, decay=0.9),
            # Generator
            linear_z0=L.Linear(dims+n_cls, 64 * 7 * 7),
            bn_linear_z0=L.BatchNormalization(64 * 7 * 7, decay=0.9),
            # Decoder
            deconv0=L.Deconvolution2D(128, 32, ksize=4, stride=2, pad=1),
            deconv1=L.Deconvolution2D(32, 1, ksize=4, stride=2, pad=1),
            bn_deconv0=L.BatchNormalization(32, decay=0.9),
        )
        self.device = device        
        self.act = act
        self.n_cls = n_cls
        self.dims = dims

    def __call__(self, x, y, z, test=False):
        bs = x.shape[0]
        sd = x.shape[2]
        
        # Encode
        y_enc = self.generate_onehotmap(bs, sd, y)
        h_enc = F.concat((x, y_enc))
        h_enc = self.conv0(h_enc)
        h_enc = self.bn_conv0(h_enc)
        h_enc = self.act(h_enc)

        h_enc = self.conv1(h_enc)
        h_enc = self.bn_conv1(h_enc)
        h_enc = self.act(h_enc)

        # Generate
        y_gen = self.generate_onehot(bs, y)
        h_gen = F.concat((z, y_gen))
        h_gen = self.linear_z0(h_gen)
        h_gen = self.bn_linear_z0(h_gen)
        h_gen = self.act(h_gen)
        h_gen = F.reshape(h_gen, (bs, 64, 7, 7))
        
        # Bottleneck
        h = F.concat((h_enc, h_gen), axis=1)

        # Decode
        h_dec = self.deconv0(h)
        h_dec = self.bn_deconv0(h_dec)
        h_dec = self.act(h_dec)

        h_dec = self.deconv1(h_dec)

        # Residual
        x_gen = h_dec + x
        return F.tanh(x_gen)

    def generate_onehot(self, bs, y_l=None):
        y = np.zeros((bs, self.n_cls))
        if y_l is not None:
            y[np.arange(bs), y_l] = 1.0
        y = y.astype(np.float32)
        y = to_device(y, self.device)
        return y

    def generate_onehotmap(self, bs, sd, y_l=None):
        y = np.zeros((bs, self.n_cls, sd, sd))
        if y_l is not None:
            for i in range(len(y)):
                y[i, y_l[i], :, :] = 1.
        y = y.astype(np.float32)
        y = to_device(y, self.device)
        return y
        
class Discriminator(Chain):
    """Image Discriminator
    """
    def __init__(self, device=None, act=F.relu):
        super(Discriminator, self).__init__(
            conv0=L.Convolution2D(1, 32, ksize=4, stride=2, pad=1),
            conv1=L.Convolution2D(32, 64, ksize=4, stride=2, pad=1),
            bn_conv0=L.BatchNormalization(32, decay=0.9),
            bn_conv1=L.BatchNormalization(64, decay=0.9),
            linear0=L.Linear(64*7*7, 32),
            linear1=L.Linear(32, 1),
            bn_linear0=L.BatchNormalization(32, decay=0.9)
        )
        self.device = device
        self.act = act

    def __call__(self, x):
        h = self.conv0(x)
        h = self.bn_conv0(h)
        h = self.act(h)
        
        h = self.conv1(h)
        h = self.bn_conv1(h)
        h = self.act(h)
        
        h = self.linear0(h)
        h = self.bn_linear0(h)
        h = self.act(h)

        h = self.linear1(h)
        return F.sigmoid(h)
