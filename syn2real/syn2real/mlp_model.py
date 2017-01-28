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
            linear0=L.Linear(784, 500, ),
            linear1=L.Linear(500, 250, ),
            linear2=L.Linear(250, 100, ),
            linear3=L.Linear(100, 10, ),
            bn0=L.BatchNormalization(500, decay=0.9),
            bn1=L.BatchNormalization(250, decay=0.9),
            bn2=L.BatchNormalization(100, decay=0.9),
        )

        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []
        
        h = self.linear0(x)
        h = self.bn0(h, test=test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h, test=test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear2(h)
        h = self.bn2(h, test=test)
        h = self.act(h)
        self.hiddens.append(h)

        y = self.linear3(h)

        return y

class Decoder(Chain):

    def __init__(self, act=F.relu):

        super(Decoder, self).__init__(
            linear0=L.Linear(10, 100, ),
            linear1=L.Linear(100, 250, ),
            linear2=L.Linear(250, 500, ),
            linear3=L.Linear(500, 784, ),
            bn0=L.BatchNormalization(100, decay=0.9),
            bn1=L.BatchNormalization(250, decay=0.9),
            bn2=L.BatchNormalization(500, decay=0.9),
        )
                
        self.act = act
        self.hiddens = []

    def __call__(self, y, test=False):
        self.hiddens = []
        
        h = self.linear0(y)
        h = self.bn0(h, test=test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h, test=test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear2(h)
        h = self.bn2(h, test=test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear3(h)
        h = F.tanh(h)

        return h
        
class Generator(Chain):

    def __init__(self, act=F.relu, dim_rand=30):
        super(Generator, self).__init__(
            linear0=L.Linear(784+dim_rand, 500, ),
            linear1=L.Linear(500, 250, ),
            linear2=L.Linear(250, 500, ),
            linear3=L.Linear(500, 784, ),
            bn0=L.BatchNormalization(500, decay=0.9),
            bn1=L.BatchNormalization(250, decay=0.9),
            bn2=L.BatchNormalization(500, decay=0.9),
        )
        
        self.act = act

    def __call__(self, x, z, test=False):
        h = F.concat((x, z))

        h = self.linear0(h)
        h = self.bn0(h, test=test)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h, test=test)
        h = self.act(h)
        
        h = self.linear2(h)
        h = self.bn2(h, test=test)
        h = self.act(h)

        h = self.linear3(h)
        h = F.tanh(h)

        return h

class Discriminator(Chain):

    def __init__(self, act=F.relu):
        super(Discriminator, self).__init__(
            linear0=L.Linear(784, 500, ),
            linear1=L.Linear(500, 250, ),
            linear2=L.Linear(250, 100, ),
            linear3=L.Linear(100, 1, ),
            bn0=L.BatchNormalization(500, decay=0.9),
            bn1=L.BatchNormalization(250, decay=0.9),
            bn2=L.BatchNormalization(100, decay=0.9),
        )
        
        self.act = act

    def __call__(self, x, test=False):
        h = self.linear0(x)
        h = self.bn0(h, test=test)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h, test=test)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h, test=test)
        h = self.act(h)

        h = self.linear3(h)
        return h

class AutoEncoder(Chain):

    def __init__(self, act=F.relu):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act)
        )
