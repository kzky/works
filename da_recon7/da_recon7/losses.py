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

class ReconstructionLoss(Chain):

    def __init__(self,
                     ):
        super(ReconstructionLoss, self).__init__()
        self.loss = None
        
    def __call__(self, x_recon, x):
        bs = x.shape[0]
        d = np.prod(x.shape[1:])
        self.loss = F.mean_squared_error(x_recon, x) / d

        return self.loss

class NegativeEntropyLoss(Chain):

    def __init__(self, test=False):
        super(NegativeEntropyLoss, self).__init__()
        self.loss = None
        
    def __call__(self, y, ):
        bs = y.data.shape[0]
        d = np.prod(y.data.shape[1:])

        y_normalized = F.softmax(y)
        y_log_softmax = F.log_softmax(y)
        self.loss = - F.sum(y_normalized * y_log_softmax) / bs / d

        return self.loss
