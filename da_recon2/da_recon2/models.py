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

class MLPBranch(Chain):
    """MLP Branch
    Enlarge random seeds to the two times as many dimensions as dimensions.
    """
    def __init__(self, dim, act=F.relu, sigma=0.03, device=None):
        super(MLPBranch, self).__init__(
            linear0=L.Linear(110, dim/2),   # 100 random seeeds and labels
            linear1=L.Linear(dim/2, dim),
            bn0=L.BatchNormalization(dim/2), 
            bn1=L.BatchNormalization(dim), 
        )
        self.act = act
        self.sigma = sigma
        self.device = device

    def concat(self, z, y=None, dim=10):
        bs = z.shape[0]
        if y is None:
            if self.device:
                y = cp.zeros((bs, dim))
            else:
                y = np.zeros((bs, dim)).astype(np.float32)
        return F.concat((z, y))

    def __call__(self, z, y=None, test=False):
        z = self.concat(z, y)

        h = self.linear0(z)
        h = self.bn0(h)
        h = self.act(h)

        h = self.linear1(h)
        h = self.bn1(h)
        return h

class MLPGenerator(Chain):
    def __init__(self, act=F.relu, sigma=0.03, device=None):
        super(MLPGenerator, self).__init__(
            linear0=L.Linear(100, 250),  
            linear1=L.Linear(250, 500),
            linear2=L.Linear(500, 750),
            linear3=L.Linear(750, 1000),
            linear4=L.Linear(1000, 784),
            bn0=L.BatchNormalization(250), 
            bn1=L.BatchNormalization(500), 
            bn2=L.BatchNormalization(750), 
            bn3=L.BatchNormalization(1000),
            branch0=MLPBranch(250, act, sigma, device),
            branch1=MLPBranch(500, act, sigma, device),
            branch2=MLPBranch(750, act, sigma, device),
            branch3=MLPBranch(1000, act, sigma, device),
            )

        self.act = act
        self.sigma = sigma
        self.device = device
        self.hiddens = []

    def generate(self, bs, dim=100):
        if self.device:
            return cp.random.randn(bs, dim) * self.sigma
        else:
            return np.random.randn(bs, dim).astype(np.float32) * self.sigma
            
    def __call__(self, bs, dim=100, y=None, test=False):
        self.hiddens = []
        z = self.generate(bs, dim)        

        # Linear/BatchNorm/Branch/Nonlinear
        h = self.linear0(z)
        h = self.bn0(h, test)
        z = self.generate(bs, dim)
        b = self.branch0(z, y, test)
        h = h + b
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        z = self.generate(bs, dim)
        b = self.branch1(z, y, test)
        h = h + b        
        h = self.act(h)
        self.hiddens.append(h)
        
        h = self.linear2(h)
        h = self.bn2(h, test)
        z = self.generate(bs, dim)
        b = self.branch2(z, y, test)
        h = h + b        
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear3(h)
        h = self.bn3(h, test)
        z = self.generate(bs, dim)
        b = self.branch3(z, y, test)
        h = h + b        
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear4(h)
        return h

class MLPEncoder(Chain):
    """Ladder-like architecture.
    """
    
    def __init__(self, act=F.relu, sigma=0.3, device=None):
        super(MLPEncoder, self).__init__(
            linear0=L.Linear(784, 1000),
            linear1=L.Linear(1000, 750),
            linear2=L.Linear(750, 500),
            linear3=L.Linear(500, 250),
            linear_mu=L.Linear(250, 100),
            linear_sigma=L.Linear(250, 100),
            bn0=L.BatchNormalization(1000),
            bn1=L.BatchNormalization(750),
            bn2=L.BatchNormalization(500), 
            bn3=L.BatchNormalization(250), 
            )

        self.act = act
        self.device = device
        self.hiddens = []
        self.mu = None
        self.log_sigma_2 = None
        self.sigma_2 = None
        self._sigma = sigma

    def generate(self, h):
        bs, dim = h.shape
        if self.device:
            return cp.random.randn(bs, dim) * self._sigma
        else:
            return np.random.randn(bs, dim).astype(np.float32) * self._sigma
        
    def __call__(self, x, test=False):
        self.hiddens = []

        # Linear/BatchNorm/Branch/Nonlinear
        h = self.linear0(x)
        h = self.bn0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear2(h)
        h = self.bn2(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear3(h)
        h = self.bn3(h, test)
        h = self.act(h)  #TODO: should use tanh?
        self.hiddens.append(h)

        # Variational
        self.mu = self.linear_mu(h)
        self.log_sigma_2 = self.linear_sigma(h)
        self.sigma_2 = F.exp(self.log_sigma_2)  #TODO: consider nan problem
        sigma = F.sqrt(self.sigma_2)
        r = self.generate(self.mu)
        z = self.mu + sigma * r
        
        return h

class MLPDecoder(Chain):
    def __init__(self, act=F.relu, device=None):
        super(MLPDecoder, self).__init__(
            linear0=L.Linear(100, 250),
            linear1=L.Linear(250, 500),
            linear2=L.Linear(500, 750),
            linear3=L.Linear(750, 1000),
            linear4=L.Linear(1000, 784),
            bn0=L.BatchNormalization(250),
            bn1=L.BatchNormalization(500),
            bn2=L.BatchNormalization(750),
            bn3=L.BatchNormalization(1000)
            )

        self.act = act
        self.device = device
        self.encoder = None
        self.hiddens = []
    
    def __call__(self, h, test=False):
        """
        Parameters
        -----------------
        h: Variable
            Shape of h is the same as that of (y; z), which is the input for Genrator.
        """
        self.hiddens = []

        # Linear/BatchNorm/Branch/Nonlinear
        h = self.linear0(h)
        h = self.bn0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear2(h)
        h = self.bn2(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear3(h)
        h = self.bn3(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear4(h)
        return h

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
        # Recon Loss for Feature
        recon_loss = 0
        if enc_hiddens is not None:
            for h0, h1 in zip(enc_hiddens[::-1], dec_hiddens):
                d = np.prod(h0.data.shape[1:])
                recon_loss += F.mean_squared_error(h0, h1) / d

        # Reconstruction Loss for Sample 
        if x_recon is not None:
            d = np.prod(x.data.shape[1:])
            recon_loss += F.mean_squared_error(x_recon, x) / d

        self.loss = recon_loss
        
        return self.loss

class VariationalLoss(Chain):
    def __init__(self, ):
        self.loss = None

    def __call__(self, mu, sigma_2, log_sigma_2):
        bs = mu.shape[0]
        return F.sum(1 + log_sigma_2 - mu**2 - sigma_2) / 2 / bs  # Explicit KL form
        
class MLPModel(Chain):
    def __init__(self, act=F.relu, sigma=0.03, device=None):
        super(MLPModel, self).__init__(
            mlp_gen = MLPGenerator(act, sigma, device),
            mlp_enc = MLPEncoder(act, sigma, device),
            mlp_dec = MLPDecoder(act, device)
        )

        self.recon_loss = ReconstructionLoss()
        self.variational_loss = VariationalLoss()

    def __call__(self, x, test=False):
        pass
        
