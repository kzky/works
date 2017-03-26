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

class ReconstructionLoss(Chain):

    def __init__(self,
                     ):
        super(ReconstructionLoss, self).__init__()
        self.loss = None
        
    def __call__(self, x_recon, x):
        bs = x.shape[0]
        d = np.prod(x.shape[1:])
        self.loss = F.mean_squared_error(x_recon, x) #/ d

        return self.loss

class ReconstructionLoss1(Chain):

    def __init__(self,
                     ):
        super(ReconstructionLoss1, self).__init__()
        self.loss = None
        
    def __call__(self, x_recon, x):
        bs = x.shape[0]
        d = np.prod(x.shape[1:])
        self.loss = F.mean_absolute_error(x_recon, x) / d

        return self.loss

class GANLoss(Chain):

    def __init__(self, ):
        super(GANLoss, self).__init__(
        )
        
    def __call__(self, d_x_gen, d_x_real=None):
        bs_d_x_gen = d_x_gen.shape[0]
        if d_x_real is not None:
            bs_d_x_real = d_x_real.shape[0]
            loss = F.sum(F.log(d_x_real)) / bs_d_x_real \
                   + F.sum(F.log(1 - d_x_gen)) / bs_d_x_gen
            return - loss  # to minimize
            
        else:
            loss = F.sum(F.log(d_x_gen)) / bs_d_x_gen
            return - loss  # to minimize (reverse trick)

class WGANLoss(Chain):
    """Wasserstein GAN loss
    """
    def __init__(self, ):
        super(WGANLoss, self).__init__(
        )
        
    def __call__(self, d_x_gen, d_x_real=None):
        bs_d_x_gen = d_x_gen.shape[0]
        if d_x_real is not None:
            bs_d_x_real = d_x_real.shape[0]
            loss = F.sum(d_x_real) / bs_d_x_real - F.sum(d_x_gen) / bs_d_x_gen
            return  - loss  # to minimize
            
        else:
            loss = F.sum(d_x_gen) / bs_d_x_gen
            return - loss  # to minimize (reverse trick)
        
class LSGANLoss(Chain):
    """Least Square GAN Loss
    """
    def __init__(self, ):
        super(LSGANLoss, self).__init__(
        )
        
    def __call__(self, d_x_gen, d_x_real=None):
        bs_d_x_gen = d_x_gen.shape[0]
        if d_x_real is not None:
            bs_d_x_real = d_x_real.shape[0]
            loss = F.sum(F.square(d_x_real - 1)) / bs_d_x_real /2 \
                   + F.sum(F.square(d_x_gen)) / bs_d_x_gen / 2
            return loss
            
        else:
            loss = F.sum(F.square(d_x_gen - 1)) / bs_d_x_gen / 2
            return loss
