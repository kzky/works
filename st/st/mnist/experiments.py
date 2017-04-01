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
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """Enc-MLP-Dec-Dis

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Losses
        self.recon_loss = ReconstructionLoss()
        self.gan_loss = GANLoss()
        self.er_loss = EntropyRegularizationLoss()

        from st.mnist.cnn_model_000 import Model

        # Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None

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

        return loss
        
    def test(self, x, y):
        y_pred = self.encoder(x, test=True)
        acc = F.accuracy(y_pred, y)
        return acc
