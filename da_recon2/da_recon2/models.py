"""Models
"""
import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer.cuda import cupy as cp
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from utils import to_device

class MLPGenerator(Chain):
    def __init__(self, act=F.relu, sigma=0.03, test=False, device=None):
        super(MLPGenerator, self).__init__(
            linear0=L.Linear(250, 20),  
            linear1=L.Linear(500, 250),
            linear2=L.Linear(750, 500),
            linear3=L.Linear(1000, 750),
            linear4=L.Linear(784, 1000),
            bn0=L.BatchNormalization(250), 
            bn1=L.BatchNormalization(500), 
            bn2=L.BatchNormalization(750), 
            bn3=L.BatchNormalization(1000)
            )

        self.act = act
        self.sigma = sigma
        self.test = test
        self.device = device

        self.hiddens = []

    def generate(self, h):
        shape = h.shape
        if self.device:
            return cp.random.randn(shape)
        else:
            return cp.random.randn(shape)

    def __call__(y, z):
        h = F.vstack([y, z])
        h = self.linear0(h)
        h = self.bn0(h)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear3(h)
        h = self.bn3(h)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear4(h)
        return h

class MLPEncoder(Chain):
    """Ladder-like architecture.
    """
    
    def __init__(self, act=F.relu, test=False, device=None):
        super(MLPEncoder, self).__init__(
            linear0=L.Linear(784, 1000),
            linear1=L.Linear(1000, 750),
            linear2=L.Linear(750, 500),
            linear3=L.Linear(500, 250),
            linear4=L.Linear(250, 20),  # bottleneck
            bn0=L.BatchNormalization(1000),
            bn1=L.BatchNormalization(750),
            bn2=L.BatchNormalization(500), 
            bn3=L.BatchNormalization(250), 
            )

        self.act = act
        self.test = test
        self.device = device
        self.generator = None
        self.hiddens = []

    def __call__(x, ):
        h = self.linear0(h)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h)
        h = self.act(h)

        h = self.linear3(h)
        h = self.bn3(h)
        h = self.act(h)

        h = self.linear4(h)
        return h
        
    def set_generator(self, generator):
        self.generator = generator

class MLPDecoder(Chain):
    def __init__(self, act=F.relu, test=False, device=None):
        super(MLPDecoder, self).__init__(
            linear0=L.Linear(250, 20),  
            linear1=L.Linear(500, 250),
            linear2=L.Linear(750, 500),
            linear3=L.Linear(1000, 750),
            linear4=L.Linear(784, 1000),
            bn0=L.BatchNormalization(250),
            bn1=L.BatchNormalization(500),
            bn2=L.BatchNormalization(750),
            bn3=L.BatchNormalization(1000)
            )

        self.act = act
        self.test = test
        self.device = device
        self.encoder = None
    
    def __call__(h):
        """
        Parameters
        -----------------
        h: Variable
            Shape of h is the same as that of (y; z), which is the input for Genrator.
        """
        h = self.linear0(h)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h)
        h = self.act(h)

        h = self.linear3(h)
        h = self.bn3(h)
        h = self.act(h)

        h = self.linear4(h)
        return h

    def set_encoder(self, encoder):
        self.encoder = encoder

class ReconstructionLoss(Chain):

    def __init__(self, ):
        pass

    def __call__(self, ):
        pass
