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

class ResConvUnit(Chain):
    def __init__(self, imap, omap, act=F.relu):
        super(ResConvUnit, self).__init__(
            conv0=L.Convolution2D(imap, omap/2, ksize=1, stride=1, pad=0, ),
            bn0=L.BatchNormalization(omap/2, decay=0.9, use_cudnn=True),
            conv1=L.Convolution2D(imap/2, omap/2, ksize=3, stride=1, pad=1, ),
            bn1=L.BatchNormalization(omap/2, decay=0.9, use_cudnn=True),
            conv2=L.Convolution2D(imap/2, omap, ksize=1, stride=1, pad=0, ),
            bn2=L.BatchNormalization(omap, decay=0.9, use_cudnn=True),
        )
        self.act = act
    
    def __call__(self, x, test=False):
        h = self.conv0(x)
        h = self.bn0(h, test)
        h = self.act(h)

        h = self.conv1(h)
        h = self.bn1(h, test)
        h = self.act(h)

        h = self.conv2(h)
        h = h + x
        h = self.bn2(h, test)
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
            convunit=ConvUnit(3, 64, k=3, s=1, p=1, act=act),
            resconv0=ResConvUnit(64, 64),
            resconv1=ResConvUnit(64, 64),
            resconv2=ResConvUnit(64, 64),
            resconv3=ResConvUnit(64, 64),
            resconv4=ResConvUnit(64, 64),
            resconv5=ResConvUnit(64, 64),
            linear=L.Linear(64*4*4, 64),
            bn=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
        )
        self.hiddens = []
        self.act = act
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.convunit(x)
        self.hiddens.append(h)

        h = self.resconv0(h)
        self.hiddens.append(h)
        h = self.resconv1(h)
        h = F.max_pooling_2d(h, (2, 2))  # 32 -> 16
        self.hiddens.append(h)

        h = self.resconv2(h)
        self.hiddens.append(h)
        h = self.resconv3(h)
        h = F.max_pooling_2d(h, (2, 2))  # 16 -> 8
        self.hiddens.append(h)

        h = self.resconv4(h)
        self.hiddens.append(h)
        h = self.resconv5(h)
        h = F.max_pooling_2d(h, (2, 2))  # 8 -> 4
        self.hiddens.append(h)

        h = self.linear(h)
        h = self.bn(h)
        return h

class MLP(Chain):
    def __init__(self, device=None, act=F.relu):
        super(MLP, self).__init__(
            linear=L.Linear(64, 10),
        )

    def __call__(self, h):
        y = self.linear(h)
        return y

class Decoder(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Decoder, self).__init__(
            # Input
            linear=L.Linear(64, 64*4*4),
            bn=L.BatchNormalization(64*4*4, decay=0.9, use_cudnn=True),
            
            # Resconv
            resconv0=ResConvUnit(64, 64),
            resconv1=ResConvUnit(64, 64),
            resconv2=ResConvUnit(64, 64),
            resconv3=ResConvUnit(64, 64),
            resconv4=ResConvUnit(64, 64),
            resconv5=ResConvUnit(64, 64),

            # Upsampling
            deconvunit0=DeconvUnit(64, 64, k=4, s=2, p=1, act=act),
            deconvunit1=DeconvUnit(64, 64, k=4, s=2, p=1, act=act),
            deconvunit2=DeconvUnit(64, 64, k=4, s=2, p=1, act=act),
            
            # Output
            conv=L.Convolution2D(64, 3, ksize=3, stride=1, pad=1, ),
        )
        self.act= act
        self.hiddens = []

    def __call__(self, h, test=False):
        self.hiddens = []

        h = self.linear(h)
        h = self.bn(h)
        h = F.reshape(h, (h.shape[0], 64, 4, 4))
        self.hiddens.append(h)

        h = self.deconvunit0(h)  # 4 -> 8
        h = self.resconv0(h)
        self.hiddens.append(h)
        h = self.resconv1(h)
        self.hiddens.append(h)

        h = self.deconvunit1(h)  # 8 -> 16
        h = self.resconv2(h)
        self.hiddens.append(h)
        h = self.resconv3(h)
        self.hiddens.append(h)

        h = self.deconvunit2(h)  # 16 -> 32
        h = self.resconv4(h)
        self.hiddens.append(h)
        h = self.resconv5(h)
        self.hiddens.append(h)

        h = self.conv(h)

        return h

