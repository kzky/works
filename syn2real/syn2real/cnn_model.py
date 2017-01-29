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
from utils import to_device
from chainer_fix import BatchNormalization

class Encoder(Chain):

    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            conv0=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            conv1=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            linear0=L.Linear(64 * 7 * 7, 64),
            linear1=L.Linear(64, 10),
            bn0=L.BatchNormalization(32, decay=0.9),
            bn1=L.BatchNormalization(64, decay=0.9),
            bn2=L.BatchNormalization(64, decay=0.9),
        )

        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.conv0(x)
        h = self.bn0(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv1(h)
        h = self.bn1(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear0(h)
        h = self.bn2(h)
        h = self.act(h)
        self.hiddens.append(h)

        y = self.linear1(h)
        
        return y

class Decoder(Chain):

    def __init__(self, act=F.relu):

        super(Decoder, self).__init__(
            linear0=L.Linear(10, 64),
            linear1=L.Linear(64, 64 * 7 * 7),
            deconv0=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            deconv1=L.Deconvolution2D(32, 1, 4, stride=2, pad=1),
            bn0=L.BatchNormalization(64, decay=0.9),
            bn1=L.BatchNormalization(64 * 7 * 7, decay=0.9),
            bn2=L.BatchNormalization(32, decay=0.9),
        )
                
        self.act = act
        self.hiddens = []

    def __call__(self, y, test=False):
        h = self.linear0(y)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)

        bs = h.shape[0]
        h = F.reshape(h, (bs, 64, 7, 7))
        
        h = self.deconv0(h)
        h = self.bn2(h)
        h = self.act(h)
        
        h = self.deconv1(h)
        x = F.tanh(h)

        return x

class Generator(Chain):

    def __init__(self, act=F.relu, dim_rand=30):
        super(Generator, self).__init__(
        )
        
        self.act = act

    def __call__(self, x, z, test=False):
        pass

class Discriminator(Chain):

    def __init__(self, act=F.relu):
        super(Discriminator, self).__init__(
        )
        
        self.act = act

    def __call__(self, x, test=False):
        pass

class AutoEncoder(Chain):

    def __init__(self, act=F.relu):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act)
        )
