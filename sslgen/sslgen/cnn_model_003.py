"""Models

Residual Connections
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
    def __init__(self, device=None, act=F.relu, ):
        super(Encoder, self).__init__(
            conv0=L.Convolution2D(1, 64, ksize=4, stride=2, pad=1, ),
            conv1=L.Convolution2D(64, 128, ksize=4, stride=2, pad=1, ),
            conv2=L.Convolution2D(128, 128, ksize=1, stride=1, pad=0, ),
            bn0=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
            bn1=L.BatchNormalization(128, decay=0.9, use_cudnn=False),
            bn2=L.BatchNormalization(128, decay=0.9, use_cudnn=False),
        )
        self.device = device        
        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.conv0(x)  # 28x28 -> 14x14
        h = self.bn0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv1(h)  # 14x14 -> 7x7
        h = self.bn1(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv2(h)  # 7x7 -> 7x7
        h = self.bn2(h, test)
        h = self.act(h)

        return h
        
class Decoder(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, ):
        super(Decoder, self).__init__(
            deconv0=L.Deconvolution2D(128, 64, ksize=4, stride=2, pad=1, ),
            deconv1=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1, ),
            bn0=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
        )
        self.device = device        
        self.act = act

    def __call__(self, h, hiddens, y=None, test=False):
        h = h + hiddens[1]  # shortcut
        h = self.deconv0(h)  # 7x7 -> 14x14
        h = self.bn0(h, test)
        h = self.act(h)

        h = h + hiddens[0]  # shortcut
        h = self.deconv1(h)  # 14x14 -> 28x28
        h = F.tanh(h)
        return h

# Alias
Generator1 = Decoder

class Generator0(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, dim=100):
        super(Generator0, self).__init__(
            linear0=L.Linear(dim, 128*7*7),
            bn0=L.BatchNormalization(128*7*7, decay=0.9, use_cudnn=False),
        )
        self.device = device        
        self.act = act
        self.dim = dim

    def __call__(self, z, test=False):
        h = self.linear0(z)
        h = self.bn0(h, test)
        h = self.act(h)

        bs = z.shape[0]
        h = F.reshape(h, (bs, 128, 7, 7))  # 7x7
        return h

class ImageDiscriminator(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, ):
        super(ImageDiscriminator, self).__init__(
            conv0=L.Convolution2D(1, 64, ksize=4, stride=2, pad=1, ),
            conv1=L.Convolution2D(64, 128, ksize=4, stride=2, pad=1, ),
            bn0=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
            bn1=L.BatchNormalization(128, decay=0.9, use_cudnn=False),
            linear0=L.Linear(128*7*7, 1),
        )
        self.device = device
        self.act = act

    def __call__(self, x, y=None, test=False):
        h = self.conv0(x)  # 28x28 -> 14x14
        h = self.bn0(h, test)
        h = self.act(h)

        h = self.conv1(h)  # 14x14 -> 7x7
        h = self.bn1(h, test)
        h = self.act(h)

        h = self.linear0(h)
        h = F.sigmoid(h)
        return h

class PatchDiscriminator(Chain, Mixin):
    def __init__(self, device=None, act=F.relu, ):
        super(PatchDiscriminator, self).__init__(
            conv0=L.Convolution2D(1, 64, ksize=7, stride=1, pad=0, ),
            conv1=L.Convolution2D(64, 32, ksize=7, stride=1, pad=0, ),
            conv2=L.Convolution2D(32, 16, ksize=7, stride=1, pad=0, ),
            conv3=L.Convolution2D(16, 1, ksize=1, stride=1, pad=0, ),
            bn0=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
            bn1=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
            bn2=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
        )
        self.device = device
        self.act = act

    def __call__(self, x, test=False):
        h = self.conv0(x)  # 28 -> 22
        h = self.bn0(h, test)
        h = self.act(h)

        h = self.conv1(h)  # 22 -> 16
        h = self.bn1(h, test)
        h = self.act(h)

        h = self.conv2(h)  # 16 -> 10
        h = self.bn2(h, test)
        h = self.act(h)

        h = self.conv3(h)  # 10 -> 10
        h = F.sigmoid(h)

        return h

class PixelDiscriminator(Chain, Mixin):
    def __init__(self, device=None, act=F.relu):
        super(PixelDiscriminator, self).__init__(

        )
        self.device = device
        self.act = act

    def __call__(self, x, y=None, test=False):
        pass
