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
            linear0=L.Linear(64*4*4, 64),
            bn0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            linear1=L.Linear(64, 10),
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

        h = self.linear0(h)
        h = self.bn0(h)
        self.hiddens.append(h)
        y = self.linear1(h)
        return y

class Decoder(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Decoder, self).__init__(
            # Input
            linear0=L.Linear(10, 64),
            bn0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            linear1=L.Linear(64, 64*4*4),
            bn1=L.BatchNormalization(64*4*4, decay=0.9, use_cudnn=True),
            
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

            # Concat
            linearconcat=L.Linear(128, 64)
            convconcat=L.Convolution2D(128, 64, 1, 1, 0),
            convconcat0=L.Convolution2D(128, 64, 1, 1, 0),
            convconcat1=L.Convolution2D(128, 64, 1, 1, 0),
            convconcat2=L.Convolution2D(128, 64, 1, 1, 0),
            convconcat3=L.Convolution2D(128, 64, 1, 1, 0),
            convconcat4=L.Convolution2D(128, 64, 1, 1, 0),
            convconcat5=L.Convolution2D(128, 64, 1, 1, 0),
            bn_linearconcat=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat1=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat2=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat3=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat4=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_convconcat5=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            
            # Output
            conv=L.Convolution2D(64, 3, ksize=3, stride=1, pad=1, ),
        )
        self.act= act
        self.hiddens = []

    def __call__(self, y, enc_hiddens, test=False):
        h = self.linear0(h)
        h = self.bn0(h)
        h = F.concat(h, enc_hiddens.pop())
        h = self.linearconcat(h)
        h = self.bn_linearconcat(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = F.reshape(h, (h.shape[0], 64, 4, 4))
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat(h)
        h = self.bn_convconcat(h)
        
        h = self.deconvunit0(h)  # 4 -> 8
        h = self.resconv0(h)
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat0(h)
        h = self.bn_convconcat0(h)
        h = self.resconv1(h)
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat1(h)
        h = self.bn_convconcat1(h)

        h = self.deconvunit1(h)  # 8 -> 16
        h = self.resconv2(h)
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat2(h)
        h = self.bn_convconcat2(h)
        h = self.resconv3(h)
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat3(h)
        h = self.bn_convconcat3(h)

        h = self.deconvunit2(h)  # 16 -> 32
        h = self.resconv4(h)
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat4(h)
        h = self.bn_convconcat4(h)
        h = self.resconv5(h)
        h = F.concat((h, enc_hiddens.pop()))
        h = self.convconcat5(h)
        h = self.bn_convconcat5(h)

        h = self.conv(h)

        return h

