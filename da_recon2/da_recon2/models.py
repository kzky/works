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

class MLPGenerator(Chain):
    def __init__(self, act=F.relu, test=False, device=None):
        super(MLPGenerator, self).__init__(
            linear0=L.Linear(250, 20),  
            linear1=L.Linear(500, 250),
            linear2=L.Linear(750, 500),
            linear3=L.Linear(1000, 750),
            linear4=L.Linear(784, 1000),
            bn1=L.BatchNormalization(250, use_gamma=False, use_beta=False),
            bn2=L.BatchNormalization(500, use_gamma=False, use_beta=False),
            bn3=L.BatchNormalization(750, use_gamma=False, use_beta=False),
            bn4=L.BatchNormalization(1000, use_gamma=False, use_beta=False),
            sb1=L.Scale(W_shape=250, bias_term=True),
            sb2=L.Scale(W_shape=500, bias_term=True),
            sb3=L.Scale(W_shape=750, bias_term=True),
            sb4=L.Scale(W_shape=1000, bias_term=True),
            )

    def __call__(y, z):

        pass

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
            bn0=L.BatchNormalization(1000, use_gamma=False, use_beta=False),
            bn1=L.BatchNormalization(750, use_gamma=False, use_beta=False),
            bn2=L.BatchNormalization(500, use_gamma=False, use_beta=False),
            bn3=L.BatchNormalization(250, use_gamma=False, use_beta=False),
            sb0=L.Scale(W_shape=1000, bias_term=True),
            sb1=L.Scale(W_shape=750, bias_term=True),
            sb2=L.Scale(W_shape=500, bias_term=True),
            sb3=L.Scale(W_shape=250, bias_term=True),
            )

        self.generator = None

    def __call__(x, ):
        pass

    def set_encoder(self, generator):
        
        self.generator = generator

class MLPDecoder(Chain):
    def __init__(self, act=F.relu, test=False, device=None):
        super(MLPDecoder, self).__init__(
            linear0=L.Linear(250, 20),  
            linear1=L.Linear(500, 250),
            linear2=L.Linear(750, 500),
            linear3=L.Linear(1000, 750),
            linear4=L.Linear(784, 1000),
            bn1=L.BatchNormalization(250, use_gamma=False, use_beta=False),
            bn2=L.BatchNormalization(500, use_gamma=False, use_beta=False),
            bn3=L.BatchNormalization(750, use_gamma=False, use_beta=False),
            bn4=L.BatchNormalization(1000, use_gamma=False, use_beta=False),
            sb1=L.Scale(W_shape=250, bias_term=True),
            sb2=L.Scale(W_shape=500, bias_term=True),
            sb3=L.Scale(W_shape=750, bias_term=True),
            sb4=L.Scale(W_shape=1000, bias_term=True),
            )
        
        self.encoder = None
    
    def __call__(h):
        """
        Parameters
        -----------------
        h: Variable
            Shape of h is the same as that of (y; z), which is the input for Genrator.
        """
        pass

    def set_encoder(self, encoder):
        
        self.encoder = encoder

