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
from recon.utils import to_device

class ConvUnit(Chain):
    def __init__(self, imap, omap, k=4, s=2, p=1, act=F.relu):
        super(ConvUnit, self).__init__(
            conv=L.Convolution2D(imap, omap, ksize=k, stride=s, pad=p, ),
            bn=L.BatchNormalization(omap, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, test=False):
        h = self.conv(h)
        h = self.bn(h, test)
        h = self.act(h)
        return h

class DeconvUnit(Chain):

    def __init__(self, imap, omap, k=4, s=2, p=1, act=F.relu):
        super(DeconvUnit, self).__init__(
            deconv=L.Deconvolution2D(imap, omap, ksize=k, stride=s, pad=p, ),
            bn=L.BatchNormalization(omap, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, test=False):
        h = self.deconv(h)
        h = self.bn(h, test)
        h = self.act(h)
        return h
        
class Encoder(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Encoder, self).__init__(
            convunit0=ConvUnit(3, 64, k=4, s=2, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=4, s=2, p=1, act=act),
            convunit2=ConvUnit(128, 256, k=4, s=2, p=1, act=act),
        )
        self.hiddens = []
        self.act = act
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.convunit0(x, test)
        self.hiddens.append(h)
        h = self.convunit1(h, test)
        self.hiddens.append(h)
        h = self.convunit2(h, test)
        return h

class MLP(Chain):
    def __init__(self, device=None, act=F.relu):
        super(MLP, self).__init__(
            linear=L.Linear(256*4*4, 10)
        )

    def __call__(self, h):
        y = self.linear(h)
        return y

class Decoder(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Decoder, self).__init__(
            deconvunit0=DeconvUnit(256, 128, k=4, s=2, p=1, act=act),
            deconvunit1=DeconvUnit(128, 64, k=4, s=2, p=1, act=act),
            deconv=L.Deconvolution2D(64, 3, ksize=4, stride=2, pad=1, ),
        )
        self.act= act
        self.hiddens = []

    def __call__(self, h, test=False):
        self.hiddens = []

        h = self.deconvunit0(h, test)
        self.hiddens.append(h)
        h = self.deconvunit1(h, test)
        self.hiddens.append(h)
        h = self.deconv(h)
        h = F.tanh(h)
        return h

class Discriminator(Chain):

    def __init__(self, device=None, act=F.relu, n_cls=10):
        super(Discriminator, self).__init__(
            convunit0=ConvUnit(3, 64, k=4, s=2, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=4, s=2, p=1, act=act),
            convunit2=ConvUnit(128, 256, k=4, s=2, p=1, act=act),
            linear=L.Linear(256*4*4 + n_cls, 1), 
        )
        self.act= act

    def __call__(self, x, y, test=False):
        h = self.convunit0(x, test)
        h = self.convunit1(h, test)
        h = self.convunit2(h, test)
        shape = h.shape
        h = F.reshape(h, (shape[0], np.prod(shape[1:])))
        h = F.concat((h, y))
        h = self.linear(h)
        h = F.sigmoid(h)
        return h
    
