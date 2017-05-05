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
from meta_st.utils import to_device

class ConvUnit(Chain):
    def __init__(self, imaps, omaps, k=4, s=2, p=1, act=F.relu):
        super(ConvUnit, self).__init__(
            conv=L.Convolution2D(imaps, omaps, ksize=k, stride=s, pad=p, nobias=True),
            bn=L.BatchNormalization(omaps, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, test=False):
        h = self.conv(h)
        h = self.bn(h, test)
        h = self.act(h)
        return h

class DeconvUnit(Chain):
    def __init__(self, imaps, omaps, k=4, s=2, p=1, act=F.relu):
        super(DeconvUnit, self).__init__(
            deconv=L.Deconvolution2D(imaps, omaps, ksize=k, stride=s, pad=p, nobias=True),
            bn=L.BatchNormalization(omaps, decay=0.9, use_cudnn=True),
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
            # ConvBlock0
            conv00=ConvUnit(3, 96, k=3, s=1, p=1),
            conv01=ConvUnit(96, 96, k=3, s=1, p=1),
            conv02=ConvUnit(96, 96, k=3, s=1, p=1),
            bn0=L.BatchNormalization(96),
            # ConvBlock1
            conv10=ConvUnit(96, 192, k=3, s=1, p=1),
            conv11=ConvUnit(192, 192, k=3, s=1, p=1),
            conv12=ConvUnit(192, 192, k=3, s=1, p=1),
            bn1=L.BatchNormalization(192),
            # ConvBlock3
            conv20=ConvUnit(192, 192, k=3, s=1, p=0),
            conv21=ConvUnit(192, 192, k=1, s=1, p=0),
            conv22=ConvUnit(192, 10, k=1, s=1, p=0),
            bn2=L.BatchNormalization(10)
        )
        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.conv00(x, test)
        h = self.conv01(h, test)
        h = self.conv02(h, test)
        h = F.max_pooling_2d(h, (2, 2))  # 32 -> 16
        h = self.bn0(h, test)
        self.hiddens.append(h)
    
        h = self.conv10(h, test)
        h = self.conv11(h, test)
        h = self.conv12(h, test)
        h = F.max_pooling_2d(h, (2, 2))  # 16 -> 8
        h = self.bn1(h, test)
        self.hiddens.append(h)
    
        h = self.conv20(h, test)  # 8 -> 6
        self.hiddens.append(h)
        h = self.conv21(h, test)
        h = self.conv22(h, test)
        h = F.average_pooling_2d(h, (6, 6))  # 6 -> 1
        h = self.bn2(h, test)
        h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
        
        return h
    
class Decoder(Chain):

    def __init__(self, device=None, act=F.relu):
        
        super(Decoder, self).__init__(
            linear=L.Linear(10, 10*6*6),
            bn=L.BatchNormalization(10*6*6),

            # ConvBlock0
            conv00=ConvUnit(10, 192, k=1, s=1, p=0),
            conv01=ConvUnit(192, 192, k=1, s=1, p=0),
            deconv02=DeconvUnit(192, 192, k=3, s=1, p=0),
            bn0=L.BatchNormalization(192),

            # ConvBlock1
            conv10=ConvUnit(192, 192, k=3, s=1, p=1),
            conv11=ConvUnit(192, 192, k=3, s=1, p=1),
            conv12=ConvUnit(192, 96, k=3, s=1, p=1),
            bn1=L.BatchNormalization(96),

            # ConvBlock3
            conv20=ConvUnit(96, 96, k=3, s=1, p=1),
            conv21=ConvUnit(96, 96, k=3, s=1, p=1),
            conv22=ConvUnit(96, 3, k=3, s=1, p=1),

            # Unpool (Deconv)
            deconv1=DeconvUnit(192, 192),
            deconv2=DeconvUnit(96, 96),
        )
        self.act = act
        self.hiddens = []

    def __call__(self, x, test=False):
        self.hiddens = []
        h = self.linear(x)  # 1 -> 6
        h = self.bn(h)
        self.hiddens.append(h)
        h = F.reshape(h, (x.shape[0], 10, 6, 6))

        h = self.conv00(h, test)
        h = self.conv01(h, test)
        h = self.deconv02(h, test) # 6 -> 8
        self.hiddens.append(h)

        h = self.bn0(h, test)
        h = self.deconv1(h, test)  # 8 -> 16
        self.hiddens.append(h)
        h = self.conv10(h, test)
        h = self.conv11(h, test)
        h = self.conv12(h, test)

        h = self.bn1(h, test)
        h = self.deconv2(h, test)  # 16 -> 32
        self.hiddens.append(h)
        h = self.conv20(h, test)  
        h = self.conv21(h, test)
        h = self.conv22(h, test)
        
        return h
