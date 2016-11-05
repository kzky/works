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

class MLPEnc(Chain):

    def __init__(self,
                     dims,
                     act=F.relu,
                     noise=False,
                     lateral=False,
                     test=False,
                     device=None):

        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        scale_biases = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_in, d_out)
            l_name = "linear-enc-{:03}".format(l)
            linears[l_name] = linear

            # Normalization and BatchCorrection
            if lateral:
                batch_norm = L.BatchNormalization(d_out, decay=0.9,
                                                      use_gamma=False, use_beta=False)
                bn_name = "bn-enc-{:03d}".format(l)
                batch_norms[bn_name] = batch_norm
                scale_bias = \
                  L.Scale(W_shape=(d_out, ), bias_term=True, bias_shape=(d_out, ))
                sb_name = "sb-enc-{:03d}".format(l)
                scale_biases[sb_name] = scale_bias
            else:
                bn_name = "bn-enc-{:03d}".format(l)
                batch_norms[bn_name] = None
                sb_name = "sb-enc-{:03d}".format(l)
                scale_biases[sb_name] = None
                
        layers.update(linears)
        layers.update(batch_norms) if lateral else None
        layers.update(scale_bias) if lateral else None
        
        super(MLPEnc, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = batch_norms
        self.scale_biases = scale_biases
        self.act = act
        self.noise = noise
        self.lateral = lateral
        self.test = test
        self.device = device
        self.hiddens = []

    def __call__(self, x):
        h = x
        self.hiddens = []
        for linear, batch_norm, scale_bias in \
          zip(self.linears.values(), self.batch_norms.values(), self.scale_biases.values()):
          
          if self.noise and not self.lateral:
              n = np.random.normal(0, 0.03, h.data.shape).astype(np.float32)
              n_ = Variable(to_device(n, self.device))
              h_ = h + n_

          h_ = linear(h)
          
          if self.lateral:
              h_ = batch_norm(h_)
              n = np.random.normal(0, 0.03, h.data.shape).astype(np.float32)
              n_ = Variable(to_device(n, self.device))
              h_ = h_ + n_
              h_ = scale_bias(h_)
              #TODO: This may change
              self.hiddens.append(h)
          h = self.act(h_)
          
        return h
  
class MLPDec(Chain):

    def __init__(self, dims, act=F.relu,
                     noise=False,
                     lateral=False,
                     test=False,
                     device=None):
        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        dims_reverse = dims[::-1]
        for l, d in enumerate(zip(dims_reverse[0:-1], dims_reverse[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_in, d_out)
            l_name = "linear-dec-{:03}".format(l)
            linears[l_name] = linear

            # Normalization and BatchCorrection
            if lateral:
                batch_norm = L.BatchNormalization(d_out, decay=0.9,
                                                      use_gamma=False, use_beta=False)
                bn_name = "bn-dec-{:03d}".format(l)
                batch_norms[bn_name] = batch_norm

            else:
                bn_name = "bn-dec-{:03d}".format(l)
                batch_norms[bn_name] = None
            
        layers.update(linears)
        layers.update(batch_norms) if lateral else None

        super(MLPDec, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = batch_norms
        self.act = act
        self.lateral = lateral
        self.test = test
        self.device = device
        self.hiddens = []
            
    def __call__(self, x):
        h = x
        self.hiddens = []
        for linear, batch_norm in zip(self.linears.values(), self.batch_norms.values()):
            h_ = linear(h)
            if self.lateral:
                h_ = batch_norm(h_)
                #TODO: This may change
                self.hiddens.append(h)
            h = self.act(h_)
                
        return h
            
class SupervizedLoss(Chain):

    def __init__(self, ):
        super(SupervizedLoss, self).__init__()
        self.loss = None
        
    def __call__(self, y, t):
        self.loss = F.softmax_cross_entropy(y, t)
        return self.loss

class ReconstructionLoss(Chain):

    def __init__(self,
                     noise=False,
                     lateral=False,
                     test=False):

        super(ReconstructionLoss, self).__init__()
        self.noise = noise
        self.lateral = lateral
        self.test = test
        
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
        if self.lateral: #TODO: do something
            pass

        # Reconstruction Loss
        recon_loss = F.mean_squared_error(x_recon, x)

        self.loss = recon_loss  #TODO: Loss add lateral recon loss
        
        return self.loss

class MLPEncDecModel(Chain):
    def __init__(self,
                     dims,
                     act=F.relu,
                     noise=False,
                     lateral=False,
                     test=False,
                     device=None):

        # Constrcut models
        mlp_enc = MLPEnc(
            dims=dims,
            act=act,
            noise=noise,
            lateral=lateral,
            test=test,
            device=device)
        mlp_dec = MLPDec(
            dims=dims,
            act=act,
            noise=noise,
            lateral=lateral,
            test=test,
            device=device)
        self.supervised_loss = SupervizedLoss()
        self.recon_loss = ReconstructionLoss()

        super(MLPEncDecModel, self).__init__(
            mlp_enc=mlp_enc,
            mlp_dec=mlp_dec)

    def __call__(self, x_l, y_l, x_u, y_u):
        pass
