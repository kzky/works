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

class MLPEncoder(Chain):

    def __init__(self, device, act=F.relu):
        super(MLPEncoder, self).__init__(
            linear0=L.Linear(784, 500),
            linear1=L.Linear(500, 250),
            classifier=L.Linear(250, 10),
            bn0=L.BatchNormalization(500, decay=0.9),
            bn1=L.BatchNormalization(250, decay=0.9),
        )
        self.device = device
        self.act = act

    def __call__(self, x, test=False):
        h = self.linear0(x)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)

        z = h
        y = self.classifier(h)
        
        return y, z
        
class MLPDecoder(Chain):

    def __init__(self, device, act=F.relu):
        super(MLPDecoder, self).__init__(
            linear0=L.Linear(250, 500),
            linear1=L.Linear(500, 750),
            bn0=L.BatchNormalization(500, decay=0.9),
            bn1=L.BatchNormalization(750, decay=0.9),
        )
        self.device = device
        self.act = act

    def __call__(self, z, test=False):
        h = self.linear0(z)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)

        return h
        
class MLPAE(Chain):

    def __init__(self, device, act=F.relu):
        mlp_encoder = MLPEncoder(device, act)
        mlp_decocer = MLPDecoder(device, act)

        super(MLPAE, self).__init__(
            mlp_encoder=mlp_encoder,
            mlp_decocer=mlp_decocer,
        )
