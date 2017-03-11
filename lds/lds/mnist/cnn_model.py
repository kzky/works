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
from lds.utils import to_devicea
from lds.chainer_fix import BatchNormalization

class Encoder(Chain):

    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            # Encoder
            conv0=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            conv1=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            linear0=L.Linear(64 * 7 * 7, 32),
            linear1=L.Linear(32, 10),
            bn0=L.BatchNormalization(32, decay=0.9, use_cudnn=False),
            bn1=L.BatchNormalization(64, decay=0.9, use_cudnn=False),
            bn2=L.BatchNormalization(32, decay=0.9, use_cudnn=False),
            # BranchNet
            linear0_bn=L.Linear(32*14*14, 10),
            linear1_bn=L.Linear(64*7*7, 10),
            linear2_bn=L.Linear(32, 10),
        )

        self.act = act
        self.hiddens = []
        self.classifiers = []
        
    def __call__(self, x, test=False):
        self.hiddens = []
        self.classifiers = []

        h = self.conv0(x)
        h = self.bn0(h, test)
        h = self.act(h)
        self.hiddens.append(h)
        cls = self.linear0_bn(h)
        self.classifiers.append(cls)

        h = self.conv1(h)
        h = self.bn1(h, test)
        h = self.act(h)
        self.hiddens.append(h)
        cls = self.linear1_bn(h)
        self.classifiers.append(cls)

        h = self.linear0(h)
        h = self.bn2(h, test)
        h = self.act(h)
        self.hiddens.append(h)
        cls = self.linear2_bn(h)
        self.classifiers.append(cls)

        h = self.linear1(h)

        return h

class Decoder(Chain):

    def __init__(self, act=F.relu):
        super(Decoder, self).__init__(
            linear0=L.Linear(10, 32),
            linear1=L.Linear(32, 64 * 7 * 7),
            deconv0=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            deconv1=L.Deconvolution2D(32, 1, 4, stride=2, pad=1),
            bn0=L.BatchNormalization(32, decay=0.9, use_cudnn=False),
            bn1=L.BatchNormalization(64 * 7 * 7, decay=0.9, use_cudnn=False),
            bn2=L.BatchNormalization(32, decay=0.9, use_cudnn=False),
        )
                
        self.act = act
        self.hiddens = []

    def __call__(self, y, test=False):
        self.hiddens = []

        h = self.linear0(y)
        h = self.bn0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        h = self.act(h)
        bs = h.shape[0]
        h = F.reshape(h, (bs, 64, 7, 7))
        self.hiddens.append(h)

        h = self.deconv0(h)
        h = self.bn2(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv1(h)
        return h


class AutoEncoder(Chain):
    
    def __init__(self, act=F.relu):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act)
        )
        


    
