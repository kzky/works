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

class MLPEnc(Chain):

    def __init__(self, dims, act=F.relu,
                     bn=True,
                     noise=False,
                     lateral=False,
                     test=False):

        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_i, d_out)
            l_name = "mlp-{:03}".format(l)
            linears[l_name] = linear

            # BatchNorm
            if bn:
                batch_norm = L.BatchNorm(d_out, decay=0.9)
                bn_name = "bn-{:03d}".format(l)
                batch_norms[bn_name] = batch_norm
            else:
                batch_norms[bn_name] = None
                
        layers.update(linears)
        layers.update(batch_norms)
        
        super(MLPEnc, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = self.batch_norms
        self.act = act
        self.bn = bn
        self.lateral = lateral
        self.test = test
        self.mid_layers = []

    def __call__(self, x):
        h = x
        self.mid_layers = []
        for linear, bath_norm in zip(self.layers.values(), self.batch_norms.value()):
            h_ = linear(h)
            if lateral:  #TODO: This may change
                self.mid_layers.append(h)
            if self.bn:
                h_ = batch_norm(h_)
            if self.lateral: #TODO: Do something
                pass
            h = self.act(h_)
                
        return h

class MLPDec(Chain):

    def __init__(self, dims, act=F.relu,
                     bn=True,
                     noise=False,
                     lateral=False,
                     test=False):
        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        dims_reverse = dims[::-1]
        for l, d in enumerate(zip(dims_reverse[0:-1], dims_reverse[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_i, d_out)
            l_name = "mlp-{:03}".format(l)
            linears[l_name] = linear

            # BatchNorm
            if bn:  #TODO: Do something if lateral is True
                batch_norm = L.BatchNorm(d_out, decay=0.9)
                bn_name = "bn-{:03d}".format(l)
                batch_norms[bn_name] = batch_norm
            else:
                batch_norms[bn_name] = None
            
        layers.update(linears)
        layers.update(batch_norms)

        super(MLPDec, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = self.batch_norms
        self.act = act
        self.bn = bn
        self.lateral = lateral
        self.test = test
        self.mid_layers = []
            
    def __call__(self, x):
        h = x
        self.mid_layers = []
        for linear, batch_norm in zip(self.layers.values(), self.batch_nomrs.values()):
            h_ = linear(h)
            if lateral:  #TODO: This may change
                self.mid_layers.append(h)
            if self.bn:
                h_ = batch_norm(h_)
            if self.lateral: #TODO: Do something
                pass
            h = self.act(h_)
                
        return h
            
class SupervizedLoss(Chain):

    def __init__(self, ):
        super(SupervizedLoss, self).__init__()
        self.acc = None
        self.loss = None
        
    def __call__(self, y, t):
        self.loss = F.softmax_cross_entropy(y, t)
        self.acc = F.accuracy(y, t)
        return self.loss

class ReconstructionLoss(Chain):

    def __init__(self,
                     bn=True,
                     noise=False,
                     lateral=False,
                     test=False):

        super(ReconstructionLoss, self).__init__()
        self.bn = bn
        self.noise = noise
        self.lateral = lateral
        self.test = test
        
        self.loss = None
        
    def __call__(self, x, mlp_enc, mlp_dec):
        """
        Parameters
        -----------------
        x: Variable
        mlp_enc: MLPEnc
        mlp_dec: MLPDec
        """

        # Lateral Recon Loss
        if lateral: #TODO: do something
            pass

        # Recon Loss
        x_recon = mlp_dec.linears[-1]
        recon_loss = mean_squared_error = (x, x_recon)

        self.loss = recon_loss  #TODO: Loss add lateral recon loss
        
        return self.loss
            
        
