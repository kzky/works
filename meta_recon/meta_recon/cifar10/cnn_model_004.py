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
from meta_recon.utils import to_device
from meta_recon.mean_only_bn import BatchNormalization

class ConvUnit(Chain):
    def __init__(self, imap, omap, k=4, s=2, p=1, act=F.relu):
        super(ConvUnit, self).__init__(
            conv=L.Convolution2D(imap, omap, ksize=k, stride=s, pad=p, ),
            bn=BatchNormalization(omap, decay=0.9, use_cudnn=True),
            bn_enc=BatchNormalization(omap, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, enc=False, test=False):
        h = self.conv(h)
        if not enc:
            h = self.bn(h, test)
        else:
            h = self.bn_enc(h, test)
        h = self.act(h)
        return h

class DeconvUnit(Chain):

    def __init__(self, imap, omap, k=4, s=2, p=1, act=F.relu):
        super(DeconvUnit, self).__init__(
            deconv=L.Deconvolution2D(imap, omap, ksize=k, stride=s, pad=p, ),
            bn=BatchNormalization(omap, decay=0.9, use_cudnn=True),
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
            convunit0=ConvUnit(3, 64, k=3, s=1, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=3, s=1, p=1, act=act),
            convunit2=ConvUnit(128, 256, k=3, s=1, p=1, act=act),
            linear=L.Linear(256*4*4, 128),
            bn=BatchNormalization(128, decay=0.9, use_cudnn=True),
        )
        self.hiddens = []
        self.act = act
        
    def __call__(self, x, enc=False, test=False):
        self.hiddens = []

        h = self.convunit0(x, enc, test)
        h = F.max_pooling_2d(h, (2, 2))
        self.hiddens.append(h)
        h = self.convunit1(h, enc, test)
        h = F.max_pooling_2d(h, (2, 2))
        self.hiddens.append(h)
        h = self.convunit2(h, enc, test)
        h = F.max_pooling_2d(h, (2, 2))
        self.hiddens.append(h)
        h = self.linear(h)
        h = self.bn(h)
        return h

class MLP(Chain):
    def __init__(self, device=None, act=F.relu):
        super(MLP, self).__init__(
            linear=L.Linear(128, 10),
        )

    def __call__(self, h):
        y = self.linear(h)
        return y

class Decoder(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Decoder, self).__init__(
            linear=L.Linear(128, 256*4*4),
            bn=L.BatchNormalization(256*4*4, decay=0.9, use_cudnn=True),
            deconvunit0=DeconvUnit(256, 128, k=4, s=2, p=1, act=act),
            deconvunit1=DeconvUnit(128, 64, k=4, s=2, p=1, act=act),
            deconv=L.Deconvolution2D(64, 3, ksize=4, stride=2, pad=1, ),
        )
        self.act= act
        self.hiddens = []

    def __call__(self, h, test=False):
        self.hiddens = []
        h = self.linear(h)
        h = self.bn(h)
        h = F.reshape(h, (h.shape[0], 256, 4, 4))
        self.hiddens.append(h)
        h = self.deconvunit0(h, test)
        self.hiddens.append(h)
        h = self.deconvunit1(h, test)
        self.hiddens.append(h)
        h = self.deconv(h)
        h = F.tanh(h)
        return h

