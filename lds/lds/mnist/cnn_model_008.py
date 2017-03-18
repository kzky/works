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
from lds.utils import to_device
from lds.chainer_fix import BatchNormalization

class Encoder(Chain):

    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            conv0=L.Convolution2D(1, 32, 3, stride=1, pad=1),
            conv1=L.Convolution2D(32, 32, 3, stride=1, pad=1),
            conv2=L.Convolution2D(32, 32, 3, stride=1, pad=1),
            conv3=L.Convolution2D(32, 64, stride=1, pad=1),
            conv4=L.Convolution2D(64, 64, 3, stride=1, pad=1),
        )

        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []
        
        return h

class Decoder(Chain):
    """Unpooling
    """
    def __init__(self, act=F.relu):
        super(Decoder, self).__init__(
            deconv4=L.Deconvolution2D(32, 32, 3, stride=1, pad=1),

        )
                
        self.act = act
        self.hiddens = []

    def __call__(self, y, test=False):
        self.hiddens = []

        return h

class AutoEncoder(Chain):
    
    def __init__(self, act=F.relu):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act)
        )
        
