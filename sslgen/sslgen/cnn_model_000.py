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

class Mixin(object):
    
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

class Encoder(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, n_cls=10):
        super(Encoder, self).__init__(
            conv0=L.Convolution2D(1, 64, ksize=4, stride=2, pad=1, ),
            conv1=L.Convolution2D(64, 128, ksize=4, stride=2, pad=1, ),
            bn0=L.BatchNormalization(64, decay=0.9),
            bn1=L.BatchNormalization(128, decay=0.9),
        )
        self.device = device        
        self.act = act
        self.n_cls = n_cls

    def __call__(self, x, test=False):
        h = self.conv0(x)  # 28x28 -> 14x14
        h = self.bn0(h)
        h = self.act(h)

        h = self.conv1(h)  # 14x14 -> 7x7
        h = self.bn1(h)
        h = self.act(h)
        return h
        
class Decoder(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, n_cls=10):
        super(Decoder, self).__init__(
            deconv0=L.Deconvolution2D(128+n_cls, 64, ksize=4, stride=2, pad=1, ),
            deconv1=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1, ),
            bn0=L.BatchNormalization(64, decay=0.9),
        )
        self.device = device        
        self.act = act
        self.n_cls = n_cls

    def __call__(self, h, y=None, test=False):
        # Concat
        bs = h.shape[0]
        sd = h.shape[2]
        y = self.generate_onehotmap(bs, sd, y)
        h = F.concat((h, y))        

        h = self.deconv0(h)  # 7x7 -> 14x14
        h = self.bn0(h)
        h = self.act(h)

        h = self.deconv1(h)  # 14x14 -> 28x28
        h = F.tanh(h)
        return h

# Alias
Generator1 = Decoder

class Generator0(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, n_cls=10, dim=100):
        super(Generator0, self).__init__(
            linear0=L.Linear(dim, 128*7*7),
            bn0=L.BatchNormalization(128*7*7, decay=0.9),
        )
        self.device = device        
        self.act = act
        self.n_cls = n_cls
        self.dim = dim

    def __call__(self, z):
        h = self.linear0(z)
        h = self.bn0(h)
        h = self.act(h)

        bs = z.shape[0]
        h = F.reshape(h, (bs, 128, 7, 7))  # 7x7
        return h

class ImageDiscriminator(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, n_cls=10):
        super(ImageDiscriminator, self).__init__(
            conv0=L.Convolution2D(1, 64, ksize=4, stride=2, pad=1, ),
            conv1=L.Convolution2D(64, 128, ksize=4, stride=2, pad=1, ),
            bn0=L.BatchNormalization(64, decay=0.9),
            bn1=L.BatchNormalization(128, decay=0.9),
            linear0=L.Linear(128*7*7 + n_cls, 1),
        )
        self.device = device
        self.act = act
        self.n_cls = n_cls

    def __call__(self, x, y=None):
        h = self.conv0(x)  # 28x28 -> 14x14
        h = self.bn0(h)
        h = self.act(h)

        h = self.conv1(h)  # 14x14 -> 7x7
        h = self.bn1(h)
        h = self.act(h)

        h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
        y = self.generate_onehot(h.shape[0], y)
        h = F.concat((h, y))

        h = self.linear0(h)
        h = F.sigmoid(h)
        return h

class PatchDiscriminator(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, n_cls=10):
        super(PatchDiscriminator, self).__init__(
            conv0=L.Convolution2D(1, 64, ksize=4, stride=2, pad=1, ),            
            bn0=L.BatchNormalization(64, decay=0.9),
            linear0=L.Linear(64*7*7+n_cls, 1),
        )
        self.device = device
        self.act = act
        self.n_cls = n_cls

    def __call__(self, x, y=None):
        h = self.conv0(x)
        h = self.bn0(h)
        h = self.act(h)

        h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
        y = self.generate_onehot(h.shape[0], y)
        h = F.concat((h, y))

        h = self.linear0(h)
        h = F.sigmoid(h)
        return h

class PixelDiscriminator(Chain, Mixin):
    def __init__(self, device=None, act=F.relu):
        super(PixelDiscriminator, self).__init__(

        )
        self.device = device
        self.act = act

    def __call__(self, x, y=None):
        pass
