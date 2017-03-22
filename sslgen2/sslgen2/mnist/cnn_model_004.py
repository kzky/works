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
from sslgen2.utils import to_device

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
            convunit0=ConvUnit(1, 64, k=4, s=2, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=4, s=2, p=1, act=act),
            linear=L.Linear(128, 128),
            bn=L.BatchNormalization(128, decay=0.9)
        )
        
    def __call__(self, x, test=False):
        h = self.convunit0(x, test)
        h = self.convunit1(h, test)
        h = self.linear(h)
        h = self.bn(h, test)
        return h

class Decoder(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Decoder, self).__init__(
            linear=L.Linear(128, 128),
            bn=L.BatchNormalization(128, decay=0.9)
            deconvunit0=DeconvUnit(128, 64, k=4, s=2, p=1, act=act),
            deconv=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1, ),
        )
        self.act= act

    def __call__(self, x, test=False):
        h = self.linear(x)
        h = self.bn(h, test)
        h = self.deconvunit0(h, test)
        h = self.deconv(h)
        h = F.tanh(h)
        return h

class Generator0(Chain):

    def __init__(self, omap, h, w, dim=100, device=None, act=F.relu,):
        super(Generator0, self).__init__(
            linear=L.Linear(dim, omap*h*w),
            bn=L.BatchNormalization(omap*h*w, use_cudnn=True)
        )
        self.act = act
        self.omap = omap
        self.h = h
        self.w = w
        
    def __call__(self, z, test=False):
        h = self.linear(z)
        h = self.bn(h, test)
        h = F.reshape(h, (h.shape[0], self.omap, self.h, self.w))
        h = self.act(h)
        return h

class Generator(Chain):

    def __init__(self, device=None, act=F.relu ,dim=100):
        super(Generator, self).__init__(
            generator0=Generator0(128, 7, 7, dim=dim+128, device=device, act=act),
            deconvunit0=DeconvUnit(256, 128, k=1, s=1, p=0, act=act),
            deconvunit1=DeconvUnit(128, 64, k=4, s=2, p=1, act=act),
            deconv=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1),
        )
        self.act= act

    def __call__(self, h_feat, z, test=False):
        z = F.concat((h_feat, z))
        h = self.generator0(z, test)
        h = self.deconvunit0(h, test)
        h = self.deconvunit1(h, test)
        h = self.deconv(h)
        h = F.tanh(h)
        return h

class Discriminator(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Discriminator, self).__init__(
            convunit0=ConvUnit(1, 64, k=4, s=2, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=4, s=2, p=1, act=act),
            linear=L.Linear(128*2, 1), 
        )
        self.act= act

    def __call__(self, x, h_feat, test=False):
        h = self.convunit0(x, test)
        h = self.convunit1(h, test)
        h = F.average_pooling_2d(h, (7, 7))
        h = F.concat((h, h_feat))
        h = self.linear(h)
        #h = F.sigmoid(h)
        return h
    
