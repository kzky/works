"""Experiments
"""
import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, serializers
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
from meta_st.cifar10.optimizers import Adam
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """
    - Stochastic Regularization
    - Resnet x 5
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, T=3):
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
        self.model_params = OrderedDict([x for x in self.model.namedparams()])
        
        # Optimizer
        self.optimizer = Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()

    def setup_meta_learners(self, ):
        #TODO: multiple layers, loss input, modification input gardient
        self.meta_learners = []
        self.opt_meta_learners = []

        # Meta-learner
        for _ in self.model_params.params():
            # meta-learner taking gradient in batch dimension
            l = L.LSTM(1, 1)
            l.to_gpu(self.device) if self.device else None
            self.meta_learners.append(l)

            # optimizer of meta-learner
            opt = optimizers.Adam()
            opt.setup(l)
            opt.use_cleargrads()
            self.opt_meta_learners.append(opt)

    def update_parameter_by_meta_learner(self, model_params):
        namedparams = model_params
        for i, elm in enumerate(namedparams.items()):  # parameter-loop
            k, p = elm

            with cuda.get_device(self.device):
                shape = p.shape
                input_ = F.expand_dims(
                    F.reshape(Variable(p.grad), (np.prod(shape), )), axis=1)
                meta_learner = self.meta_enc_learners[i]
                g_t = meta_learner(input_)  # forward of meta-learner
                p.data -= g_t.data.reshape(shape)

                # Set parameter as variable to be backproped
                if self.t  ==  self.T:
                    w = p - F.reshape(g_t, shape)
                    self.model_params[k] = w

    def train(self, x_l0, x_l1, y_l, x_u0, x_u1):
        self.t += 1

        # Train meta-learner
        if self.t > self.T:
            self._train_meta_learner(x_l0, y_l)
            self.t = 0
            return
        
        # Train learner
        self._train(x_l0, x_l1, y_l)
        self._train(x_u0, x_u1, None)

    def _train_meta_learner(self, x, y):
        # Cross Entropy Loss
        y_pred = self.model(x, self.model_params)
        loss_ce = F.softmax_cross_entropy(y_pred, y)
        loss_ce.backward()

        loss_ce.unchain_backward()  #TODO: here is a proper place to unchain?
        for opt in self.opt_meta_learners:
            opt.update()

    def _train(self, x0, x1, y=None):
        # Cross Entropy Loss
        y_pred0 = self.model(x0, self.model_params)

        if y is not None:
            loss_ce = F.softmax_cross_entropy(y_pred0, y)
            self.cleargrads()
            loss_ce.backward()

            # update learner using loss_ce
            self.optimizer.update(self.model_params)
            return
        
        # Stochastic Regularization (i.e, Consistency Loss)
        y_pred1 = self.model(x1, self.model_params)
        loss_rec = self.recon_loss(F.softmax(y_pred0), F.softmax(y_pred1))  
        self.cleargrads()
        loss_rec.backward()

        # update learner using loss_rec and meta-learner
        update_parameter_by_meta_learner(self.model_params)
                        
    def test(self, x, y):
        y_pred = self.model(x, self.model_params, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

    def cleargrads(self, ):
        for k, v in self.model_params:
            v.cleargrad()
        
