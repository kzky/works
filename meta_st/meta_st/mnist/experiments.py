"""Experiments
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
import time
import os
import cv2
import shutil
import csv
from meta_st.utils import to_device
from meta_st.losses import ReconstructionLoss, LSGANLoss, GANLoss, EntropyRegularizationLoss
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """
    - Stochastic Regularization
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Loss
        self.recon_loss = ReconstructionLoss()

        # Model
        from meta_st.mnist.cnn_model_000 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()        

    def train(self, x_l, y_l, x_u):
        self._train(x_l, y_l)
        self._train(x_l, None)

    def _train(self, x, y=None):
        loss = 0

        # Cross Entropy Loss
        y_pred0 = self.model(x)
        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            loss += loss_ce

        # Stochastic Regularization
        y_pred1 = self.model(x)
        loss_rec = self.recon_loss(F.softmax(y_pred0), F.softmax(y_pred1))
        loss += loss_rec

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()
        
    def test(self, x, y):
        y_pred = self.model(x, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

class Experiment001(Experiment000):
    """
    - Stochastic Regularization
    - ResNet x 2
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Loss
        self.recon_loss = ReconstructionLoss()

        # Model
        from meta_st.mnist.cnn_model_001 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()        
    

class Experiment002(Experiment001):
    """
    - Stochastic Regularization
    - ResNet x 2
    - Entropy Regularization
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        super(Experiment002, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act,
            n_cls=n_cls
        )
        
        # Loss
        self.recon_loss = ReconstructionLoss()
        self.er_loss = EntropyRegularizationLoss()
        
    def _train(self, x, y=None):
        loss = 0

        # Cross Entropy Loss
        y_pred0 = self.model(x)
        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            loss += loss_ce

        # Stochastic Regularization
        y_pred1 = self.model(x)
        loss_rec = self.recon_loss(F.softmax(y_pred0), F.softmax(y_pred1))
        loss += loss_rec

        # Entropy Regularization
        loss_er0 = self.er_loss(y_pred0)
        loss_er1 = self.er_loss(y_pred1)
        loss += loss_er0 + loss_er1

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

class Experiment003(Experiment002):
    """
    - Stochastic Regularization without softmax
    - ResNet x 2
    - Entropy Regularization
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        super(Experiment003, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act,
            n_cls=n_cls
        )
        
        # Loss
        self.recon_loss = ReconstructionLoss()
        self.er_loss = EntropyRegularizationLoss()
        
    def _train(self, x, y=None):
        loss = 0

        # Cross Entropy Loss
        y_pred0 = self.model(x)
        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            loss += loss_ce

        # Stochastic Regularization
        y_pred1 = self.model(x)
        loss_rec = self.recon_loss(y_pred0, y_pred1)
        loss += loss_rec

        # Entropy Regularization
        loss_er0 = self.er_loss(y_pred0)
        loss_er1 = self.er_loss(y_pred1)
        loss += loss_er0 + loss_er1

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

class Experiment004(Experiment003):
    """
    - Stochastic Regularization b/w hiddens
    - ResNet x N
    - Entropy Regularization
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        super(Experiment004, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act,
            n_cls=n_cls
        )
        # Model
        from meta_st.mnist.cnn_model_002 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()        
    
    def _train(self, x, y=None):
        loss = 0

        # Cross Entropy Loss
        y_pred0 = self.model(x)
        hiddens0 = self.model.hiddens
        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            loss += loss_ce

        # Stochastic Regularization
        y_pred1 = self.model(x)
        hiddens1 = self.model.hiddens
        loss_rec = reduce(lambda u, v: u+v,
                          [self.recon_loss(u, v) for u, v in zip(hiddens0, hiddens0)])
        loss += loss_rec

        # Entropy Regularization
        loss_er0 = self.er_loss(y_pred0)
        loss_er1 = self.er_loss(y_pred1)
        loss += loss_er0 + loss_er1

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()
