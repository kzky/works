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
    def __init__(self, act=F.relu, sigma=0.03, device=None):
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
        self.device = device

        self.hiddens = []

    def generate(self, h):
        shape = h.shape
        if self.device:
            return cp.random.randn(shape)
        else:
            return cp.random.randn(shape)

    def __call__(y, z, test=False):
        h = F.vstack([y, z])
        h = self.linear0(h)
        h = self.bn0(h, test)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h, test)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear3(h)
        h = self.bn3(h, test)
        h = h + self.gerate(h)
        h = self.act(h)

        h = self.linear4(h)
        return h

class MLPEncoder(Chain):
    """Ladder-like architecture.
    """
    
    def __init__(self, act=F.relu, device=None):
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
        self.device = device
        self.generator = None
        self.hiddens = []

    def __call__(x, test=False):
        h = self.linear0(h)
        h = self.bn0(h, test)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h, test)
        h = self.act(h)

        h = self.linear3(h)
        h = self.bn3(h, test)
        h = self.act(h)

        h = self.linear4(h)
        return h
        
    def set_generator(self, generator):
        self.generator = generator

class MLPDecoder(Chain):
    def __init__(self, act=F.relu, device=None):
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
        self.device = device
        self.encoder = None
    
    def __call__(h, test=False):
        """
        Parameters
        -----------------
        h: Variable
            Shape of h is the same as that of (y; z), which is the input for Genrator.
        """
        h = self.linear0(h)
        h = self.bn0(h, test)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        h = self.act(h)

        h = self.linear2(h)
        h = self.bn2(h, test)
        h = self.act(h)

        h = self.linear3(h)
        h = self.bn3(h, test)
        h = self.act(h)

        h = self.linear4(h)
        return h

    def set_encoder(self, encoder):
        self.encoder = encoder

class ReconstructionLoss(Chain):

    def __init__(self, ):
        self.loss = None

    def __call__(self, x_recon, x, enc_hiddens, dec_hiddens):
        """
        Parameters
        -----------------
        x_recon: Variable to be reconstructed as label
        x: Variable to be reconstructed as label
        enc_hiddens: list of Variable
        dec_hiddens: list of Varialbe
        """
        # Lateral Recon Loss
        recon_loss = 0
        if self.rc and enc_hiddens is not None:
            for h0, h1 in zip(enc_hiddens[::-1], dec_hiddens):
                d = np.prod(h0.data.shape[1:])
                recon_loss += F.mean_squared_error(h0, h1) / d

        # Reconstruction Loss
        if x_recon is not None:
            d = np.prod(x.data.shape[1:])
            recon_loss += F.mean_squared_error(x_recon, x) / d

        self.loss = recon_loss
        
        return self.loss
        
