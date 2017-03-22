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
        )
        
    def __call__(self, x, test=False):
        h = self.convunit0(x, test)
        h = self.convunit1(h, test)
        return h

class Generator0(Chain):

    def __init__(self, dim=100, device=None, act=F.relu,):
        super(Generator0, self).__init__(
            linear=L.Linear(dim, 128*7*7),
            bn=L.BatchNormalization(128*7*7, use_cudnn=True)
        )
        self.act = act
        
    def __call__(self, z, test=False):
        h = self.linear(z)
        h = self.bn(h, test)
        h = F.reshape(h, (h.shape[0], self.omap, self.h, self.w))
        h = self.act(h)
        return h

class Generator(Chain):

    def __init__(self, device=None, act=F.relu ,dim=100):
        super(Generator, self).__init__(
            deconvunit0=DeconvUnit(256, 128, k=1, s=1, p=0, act=act),
            deconvunit1=DeconvUnit(128, 64, k=4, s=2, p=1, act=act),
            deconv=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1),
        )
        self.act= act

    def __call__(self, h, hz, test=False):
        h = F.concat((h, hz))
        h = self.deconvunit0(h, test)
        h = self.deconvunit1(h, test)
        h = self.deconv(h)
        h = F.tanh(h)
        return h

Decoder = Generator
    
class Discriminator(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Discriminator, self).__init__(
            convunit0=ConvUnit(1, 64, k=4, s=2, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=4, s=2, p=1, act=act),
            conv_finetune=L.Convolution2D(256, 128, ksize=1, stride=1, pad=0),
            bn_finetune=L.BatchNormalization(128, decay=0.9, use_cudnn=True),
            linear=L.Linear(128, 1), 
        )
        self.act= act

    def __call__(self, x, h_feat, test=False):
        h = self.convunit0(x, test)
        h = self.convunit1(h, test)
        h = F.concat((h, h_feat))
        h = self.conv_finetune(h)
        h = self.bn_finetune(h, test)
        h = F.average_pooling_2d(h, (7, 7))
        h = self.linear(h)
        #h = F.sigmoid(h)
        return h
    
