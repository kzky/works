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
from meta_st.optimizers import Adam
from sklearn.metrics import confusion_matrix

class AdamLearner(Link):
    def __init__(self, dim):
        super(AdamLearner, self).__init__(
            beta1=(dim, ),
            beta2=(dim, )
        )
        self.beta1.data.fill(-1e12)
        self.beta2.data.fill(-1e12)

        self.m = Variable(np.zeros_like(self.beta1.data))
        self.v = Variable(np.zeros_like(self.beta2.data))

    def to_gpu(self, device=None):
        super(AdamLearner, self).to_gpu()

        self.m.to_gpu(device)
        self.v.to_gpu(device)

    def __call__(self, x):
        f1 = F.sigmoid(self.beta1)
        f2 = F.sigmoid(self.beta2)
        #self.m = f1 * self.m + (1 - f1) * x
        #self.v = f2 * self.v + (1 - f2) * x**2
        self.m = self.beta1 * self.m + (1 - self.beta1) * x
        self.v = self.beta2 * self.v + (1 - self.beta2) * x**2
        g = 1e-3 * self.m / F.sqrt(self.v + 1e-8)
        return g

class MetaLearner(Chain):
    def __init__(self, dim):
        super(MetaLearner, self).__init__(
            ml0=AdamLearner(dim),
        )

    def __call__(self, h):
        return self.ml0(h)

class Experiment000(object):
    """
    FCNN model
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
        from meta_st.cifar10.cnn_model_001 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None
        self.model_params = OrderedDict([x for x in self.model.namedparams()])
        
        # Optimizer
        self.optimizer = Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
        self.setup_meta_learners()

    def setup_meta_learners(self, ):
        #TODO: multiple layers, modification of inputs
        self.meta_learners = []
        self.opt_meta_learners = []

        # Meta-learner
        for k, v in self.model_params.items():
            # meta-learner taking gradient in batch dimension
            ml = MetaLearner(np.prod(v.shape))
            ml.to_gpu(self.device) if self.device is not None else None
            self.meta_learners.append(ml)

            # optimizer of meta-learner
            opt = optimizers.Adam(1e-3)
            opt.setup(ml)
            opt.use_cleargrads()
            self.opt_meta_learners.append(opt)        
                
    def train(self, x_l0, x_l1, y_l, x_u0, x_u1):
        # Train learner and meta-learner
        self._train(x_l0, x_l1, y_l)
        self._train_meta_learner(x_l0, x_l1, y_l, x_u0, x_u1)

    def _train(self, x0, x1, y):
        # Cross Entropy Loss
        y_pred0 = self.model(x0, self.model_params)
        loss_ce = F.softmax_cross_entropy(y_pred0, y)
        self.cleargrads()
        loss_ce.backward()

        self.optimizer.update(self.model_params)

    def _train_meta_learner(self, x_l0, x_l1, y_l, x_u0, x_u1):
        # Stochastic Regularization (i.e, Consistency Loss)
        y_pred0 = self.model(x_u0, self.model_params)
        y_pred1 = self.model(x_u1, self.model_params)
        #TODO: better to not use softmax?
        loss_rec = self.recon_loss(F.softmax(y_pred0), F.softmax(y_pred1))
        self.cleargrads()
        loss_rec.backward()

        # update learner using loss_rec and meta-learner
        self.update_parameter_by_meta_learner(
            self.model_params, loss_rec, 
            x_l0, x_l1, y_l)
        self.train_meta_learner(x_l1, y_l)

    def update_parameter_by_meta_learner(
            self, model_params, loss, 
            x_l0, x_l1, y_l):

        # Forward meta-learner
        namedparams = model_params
        for i, elm in enumerate(namedparams.items()):  # parameter-loop
            k, p = elm
            with cuda.get_device_from_id(self.device):
                shape = p.shape
                xp = cuda.get_array_module(p.data)

                x = p.grad
                grad = xp.reshape(x, (np.prod(shape), ))
                meta_learner = self.meta_learners[i]
                g = meta_learner(Variable(grad))  # forward
                w = p - F.reshape(g, shape)
                self.model_params[k] = w

    def train_meta_learner(self, x_l1, y_l):
        # Train meta-learner with main objective
        y_pred = self.model(x_l1, self.model_params)

        #TODO: recurrent training ends in memory leak
        loss_ml = F.softmax_cross_entropy(y_pred, y_l)
        
        self.cleargrads()  # need to clear W'grad due to loss_rec.backward
        for meta_learner in self.meta_learners:
            meta_learner.cleargrads()
        loss_ml.backward(retain_grad=True)
        for opt in self.opt_meta_learners:
            opt.update()

        loss_ce.unchain_backward()  #TODO: here is a proper place to unchain?

    def test(self, x, y):
        y_pred = self.model(x, self.model_params, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

    def cleargrads(self, ):
        for k, v in self.model_params.items():
            v.cleargrad()
        
