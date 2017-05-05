"""Experiments
"""
import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, serializers, optimizers
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
from meta_st.cifar10.datasets import Cifar10DataReader
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """
    - ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.leaky_relu):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.T = T
        self.t = 0

        # Loss
        self.recon_loss = ReconstructionLoss()

        # Model
        from meta_st.cifar10.cnn_model_000 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
        self.setup_meta_learners()

    def train(self, x, y):
        # Cross Entropy Loss
        y_pred0 = self.model(x)
        self.model.cleargrads()
        loss = F.softmax_cross_entropy(y_pred0, y)
        loss.backward()
        self.optimizer.update()
        
    def test(self, x, y):
        y_pred = self.model(x, test=True)
        acc = F.accuracy(y_pred, y)
        return acc
