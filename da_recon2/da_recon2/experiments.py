"""Experiments
"""
from da_recon2.models import MLPModel

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

class Experiment(object):
    """Experiment takes responsibility for a batch not for train-loop.
    """
    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 act=F.relu,
                 sigma=0.3,
                 ncls=10,
                 dim=100,
                 noise=False,
                 rc_feet="Gen-Enc",
                 rc_sample="No",
                 gan_loss="Gen-Enc"):

        # Setting
        self.device = device
        self.act = act
        self.sigma = sigma
        self.dim = dim
        self.noise = noise
        self.rc_feet = rc_feet
        self.rc_sample = rc_sample
        self.gan_loss = gan_loss

        # Model
        self.model = MLPModel(ncls, act=act, sigma=sigma, device=device)
        self.model.to_gpu(self.device) if self.device else None
        self.mlp_gen = self.model.mlp_gen
        self.mlp_dec = self.model.mlp_dec
        self.mlp_enc = self.model.mlp_enc
        self.recon_loss = self.model.recon_loss
        self.variational_loss = self.model.variational_loss
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()

        # Loss
        self.v_loss = None
        self.recon_loss = None
        self.recon_feet_loss = None

    def forward(self, x, y=None, test=False):
        # Encode
        z = self.mlp_enc(x)

        # Decode
        x_recon = self.mlp_dec(z)

        # Generate
        bs = x.shape[0]
        x_gen = self.mlp_gen(bs, self.dim, y, test)

        # Loss
        l = self.forward_for_loss(x, x_recon, x_gen)

        return l

    def train(self, x_l, y_l, x_u, y_u=None, test=False):
        # Forward
        loss_l = self.forward(x_l, y_l, test=False)
        loss_u = self.forward(x_u, y_u, test=False)
        loss = loss_l + loss_u

        # Backward
        self.model.cleargrads()
        loss.backward()

        #Update
        self.optimizer.update()

    def generate(self, bs, dim):
        return self.mlp_gen(bs, dim, y=None, test=True)
        
    def forward_for_loss(self, x, x_recon, x_gen):
        # Variational Loss
        v_loss = self.variational_loss(self.mlp_enc.mu,
                                       self.mlp_enc.sigma_2, self.mlp_enc.log_sigma_2)
        self.v_loss = cuda.to_cpu(v_loss.data)
        
        # Recon Loss for sample
        recon_loss = \
                     self.recon_loss(x_recon, x, None, None) \
                     + self.recon_loss(x_gen, x, None, None)
        self.recon_loss = cuda.to_cpu(recon_loss.data)
        
        # Recon Loss for feature
        enc_hiddens = self.mlp_enc.hiddens
        dec_hiddens = self.mlp_dec.hiddens
        gen_hiddens = self.mlp_gen.hiddens
        recon_feet_loss = \
                          self.recon_loss(None, None, enc_hiddens, dec_hiddens) \
                          + self.recon_loss(None, None, enc_hiddens, gen_hiddens)
        self.recon_feet_loss = cuda.to_cpu(recon_feet_loss.data)
        
        loss = recon_loss + recon_feet_loss
        return loss


# Alias
Experiment000 = Experiment
