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
from st.utils import to_device


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
            conv0=L.Convolution2D(imap, omap, ksize=1, stride=1, pad=0, ),
            bn0=L.BatchNormalization(omap, decay=0.9, use_cudnn=True),
            conv1=L.Convolution2D(imap, omap, ksize=3, stride=1, pad=1, ),
            bn1=L.BatchNormalization(omap, decay=0.9, use_cudnn=True),
            conv2=L.Convolution2D(imap, omap, ksize=1, stride=1, pad=0, ),
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

class Model(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Model, self).__init__(
            convunit=ConvUnit(1, 64, k=3, s=1, p=1, act=act),
            resconvunit0=ResConvUnit(64, 64),
            resconvunit1=ResConvUnit(64, 64),
            linear=L.Linear(64*7*7, 10),
        )
        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.convunit(x, test)
        h = F.dropout(h, train=not test)

        h = self.resconvunit0(h, test)
        self.hiddens.append(h)
        h = F.max_pooling_2d(h, (2, 2))
        h = F.dropout(h, train=not test)
        h = self.resconvunit1(h, test)
        self.hiddens.append(h)
        h = F.max_pooling_2d(h, (2, 2))
        h = F.dropout(h, train=not test)

        h = self.linear(h)
        return h

