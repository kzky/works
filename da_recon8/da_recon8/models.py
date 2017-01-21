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
            linear0=L.Linear(784, 1000),
            linear1=L.Linear(1000, 500),
            linear2=L.Linear(500, 200),
            classifier=L.Linear(200, 10),
            bn0=L.BatchNormalization(1000, decay=0.9),
            bn1=L.BatchNormalization(500, decay=0.9),
            bn2=BatchNormalization(200, decay=0.9, 
                                   use_gamma=False, use_beta=False),
            sc2=L.Scale(W_shape=(200, ), bias_term=True)
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

        h = self.linear2(h)
        h = self.bn2(h)
        z = h

        h = self.sc2(h)
        y = self.classifier(h)
        
        return y, z
        
class MLPDecoder(Chain):

    def __init__(self, device, act=F.relu):
        super(MLPDecoder, self).__init__(
            linear0=L.Linear(210, 500),
            linear1=L.Linear(500, 1000),
            linear2=L.Linear(1000, 784),
            sc=L.Scale(W_shape=(210, ), bias_term=True),
            bn0=L.BatchNormalization(500, decay=0.9),
            bn1=L.BatchNormalization(1000, decay=0.9),
        )
        self.device = device
        self.act = act

    def __call__(self, y, z, test=False):
        #TODO: Variable y should be copyed for not backpropping
        h = F.concat((y, z))  # z was normalized by whitening (BatchNorm)
        h = self.sc(h)

        h = self.linear0(h)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)

        h = self.linear2(h)
                
        return h
        
class MLPAE(Chain):

    def __init__(self, device, act=F.relu):
        mlp_encoder = MLPEncoder(device, act)
        mlp_decoder = MLPDecoder(device, act)

        super(MLPAE, self).__init__(
            mlp_encoder=mlp_encoder,
            mlp_decoder=mlp_decoder,
        )
