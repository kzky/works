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
            linear2=L.Linear(500, 250),
            linear3=L.Linear(250, 10),
            bn0=L.BatchNormalization(1000, decay=0.9),
            bn1=L.BatchNormalization(500, decay=0.9),
            bn2=L.BatchNormalization(250, decay=0.9),
        )
        self.device = device
        self.act = act
        self.hiddens = []

    def __call__(self, x, test=False):
        self.hiddens = []

        h = self.linear0(x)
        h = self.bn0(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear2(h)
        h = self.bn2(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear3(h)

        return h
        
class MLPDecoder(Chain):

    def __init__(self, device, act=F.relu):
        super(MLPDecoder, self).__init__(
            linear0=L.Linear(10, 250),
            linear1=L.Linear(500, 500),
            linear2=L.Linear(1000, 1000),
            linear3=L.Linear(2000, 784),
            bn0=L.BatchNormalization(250, decay=0.9),
            bn1=L.BatchNormalization(500, decay=0.9),
            bn2=L.BatchNormalization(1000, decay=0.9),
        )
        self.device = device
        self.act = act

    def __call__(self, x, hiddens, test=False):
        hiddens.reverse()
        
        h = self.linear0(x)
        h = self.bn0(h)
        h = self.act(h)
        h_lateral = self.normalize_linearly(hiddens[0])
        h = F.concat((h, h_lateral))

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)
        h_lateral = self.normalize_linearly(hiddens[1])
        h = F.concat((h, h_lateral))

        h = self.linear2(h)
        h = self.bn2(h)
        h = self.act(h)
        h_lateral = self.normalize_linearly(hiddens[2])
        h = F.concat((h, h_lateral))

        h = self.linear3(h)

        return h

    def normalize_linearly(self, h):
        """Normalize h linearly over dimensions in [0, 1]
        """
        h_max = F.max(h, axis=1, keepdims=True)
        h_min = F.min(h, axis=1, keepdims=True)
        h_norm = (h - h_min) / (h_max - h_min)
        
        return h_norm
        
        
class MLPAE(Chain):

    def __init__(self, device, act=F.relu):
        mlp_encoder = MLPEncoder(device, act)
        mlp_decocer = MLPDecoder(device, act)

        super(MLPAE, self).__init__(
            mlp_encoder=mlp_encoder,
            mlp_decocer=mlp_decocer,
        )
