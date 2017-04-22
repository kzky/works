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

class Model(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Model, self).__init__(
            convunit0=ConvUnit(1, 64, k=3, s=1, p=1, act=act),
            convunit1=ConvUnit(64, 128, k=3, s=1, p=1, act=act),
            linear=L.Linear(128*7*7, 10)
        )
        self.hiddens = []
        self.act = act
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.convunit0(x, test)
        h = F.max_pooling_2d(h, (2, 2))
        h = F.dropout(h, train=not test)
        h = self.convunit1(h, test)
        h = F.max_pooling_2d(h, (2, 2))
        h = F.dropout(h, train=not test)
        h = self.linear(h)
        return h

