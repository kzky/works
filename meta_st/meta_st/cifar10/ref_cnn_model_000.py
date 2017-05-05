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
from meta_st.linear import Linear
from meta_st.convolution import Convolution2D
from meta_st.deconvolution import Deconvolution2D
from meta_st.batch_normalization import BatchNormalization

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

class Model(Chain):

    def __init__(self, device=None, act=F.relu):
        
        super(Model, self).__init__(
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

    def __call__(self, x, test=False):
        h = self.conv00(x, test)
        h = self.conv01(h, test)
        h = self.conv02(h, test)
        h = F.max_pooling_2d(h, (2, 2))  # 32 -> 16
        h = self.bn0(h, test)
    
        h = self.conv10(h, test)
        h = self.conv11(h, test)
        h = self.conv12(h, test)
        h = F.max_pooling_2d(h, (2, 2))  # 16 -> 8
        h = self.bn1(h, test)
    
        h = self.conv20(h, test)  # 8 -> 6
        h = self.conv21(h, test)
        h = self.conv22(h, test)
        h = F.average_pooling_2d(h, (6, 6))  # 6 -> 1
        h = self.bn2(h, test)
        h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
        
        return h
    
