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
from st.utils import to_device
from st.losses import ReconstructionLoss, LSGANLoss, GANLoss, EntropyRegularizationLoss
from st.cifar10.datasets import Cifar10DataReader
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """
    - Stochastic Regularization
    - Resnet x N
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
        from st.cifar10.cnn_model_000 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()

    def train(self, x_l0, x_l1, y_l, x_u0, x_u1):
        self._train(x_l0, x_l1, y_l)
        self._train(x_u0, x_u1, None)

    def _train(self, x0, x1, y=None):
        loss = 0

        # Cross Entropy Loss
        y_pred0 = self.model(x0)
        y_pred1 = self.model(x1)
        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            loss += loss_ce

        # Stochastic Regularization
        loss_rec = self.recon_loss(F.softmax(y_pred0), F.softmax(y_pred1))
        loss += loss_rec

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()
        
    def test(self, x, y):
        y_pred = self.model(x, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

class Experiment001(object):
    """
    - Stochastic Regularization
    - Net in tempens (conv -> nin -> linear)
    - Mean-only BN
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
        from st.cifar10.cnn_model_001 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()

    def train(self, x_l0, x_l1, y_l, x_u0, x_u1):
        self._train(x_l0, x_l1, y_l)
        self._train(x_u0, x_u1, None)

    def _train(self, x0, x1, y=None):
        loss = 0

        # Cross Entropy Loss
        y_pred0 = self.model(x0)
        y_pred1 = self.model(x1)
        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            loss += loss_ce

        # Stochastic Regularization
        loss_rec = self.recon_loss(F.softmax(y_pred0), F.softmax(y_pred1))
        loss += loss_rec

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()
        
    def test(self, x, y):
        y_pred = self.model(x, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

class Experiment002(Experiment000):
    """
    - ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
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
        from st.cifar10.cnn_model_002 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()

class Experiment003(Experiment000):
    """
    - ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
    - with large maps
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
        from st.cifar10.cnn_model_003 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
