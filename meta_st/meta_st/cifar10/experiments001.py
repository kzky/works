"""Experiments using 
- Stochastic Regularization
- FCCN
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

class MetaLearner(Chain):
    def __init__(self, inmap=1, midmap=1, outmap=1):
        super(MetaLearner, self).__init__(
            #TODO: check initialization
            l0=L.LSTM(inmap, midmap, 
                      forget_bias_init=lambda x: 1e12*x,
                      lateral_init=lambda x: 1e-12*x,
                      upward_init=lambda x: 1e-12*x
                      ),
            l1=L.LSTM(midmap, outmap,
                      forget_bias_init=lambda x: 1e12*x,
                      lateral_init=lambda x: 1e-12*x,
                      upward_init=lambda x: 1e-12*x
                      ),
        )

    def __call__(self, h):
        h = self.l0(h)
        h = self.l1(h)
        return h
        
class Experiment000(object):
    """
    - Stochastic Regularization
    - FCCN
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.leaky_relu, T=3):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.T = T
        self.t = 0
        self.loss_ml = 0

        # Loss
        self.rc_loss = ReconstructionLoss()

        # Model
        from meta_st.cifar10.cnn_model_001 import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None
        self.model_params = OrderedDict([x for x in self.model.namedparams()])
        
        # Optimizer
        self.optimizer = Adam(learning_rate)  #TODO: adam is appropriate?
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
        self.setup_meta_learners()

    def setup_meta_learners(self, ):
        self.meta_learners = []
        self.ml_optimizers = []

        # Meta-learner
        for _ in self.model_params:
            # meta-learner taking gradient in batch dimension
            ml = MetaLearner(inmap=1, midmap=1, outmap=1)
            ml.to_gpu(self.device) if self.device is not None else None
            self.meta_learners.append(ml)

            # optimizer of meta-learner
            opt = optimizers.Adam(1e-3)
            opt.setup(ml)
            opt.use_cleargrads()
            self.ml_optimizers.append(opt)        

    def train(self, x_l0, x_l1, y_l, x_u0, x_u1):
        self._train_for_primary_task(x_l0, y_l)
        self._train_for_auxiliary_task(x_l0, x_l1, y_l, x_u0, x_u1)
        
        self.t += 1
        if self.t == self.T:
            self._train_meta_learners()
            self.t = 0

    def _train_for_primary_task(self, x_l0, y_l):
        y_pred = self.model(x_l0, self.model_params)
        loss_ce = F.softmax_cross_entropy(y_pred, y_l)
        self._cleargrads()
        loss_ce.backward()
        self.optimizer.update(self.model_params)
        
    def _train_for_auxiliary_task(self, x_l0, x_l1, y_l, x_u0, x_u1):
        # Compute gradients
        y_pred0 = self.model(x_u0, self.model_params)
        y_pred1 = self.model(x_u1, self.model_params)
        loss_rc = self.rc_loss(y_pred0, y_pred1)
        self._cleargrads()
        loss_rc.backward()

        # Update optimizee parameters
        model_params = self.model_params
        for i, elm in enumerate(model_params.items()):
            name, w = elm
            meta_learner = self.meta_learners[i]
            ml_optimizer = self.ml_optimizers[i]
            shape = w.shape
            with cuda.get_device_from_id(self.device):
                xp = cuda.get_array_module(w.data)
                grad_data = xp.reshape(w.grad, (np.prod(shape), 1))
                
            # refine grad, update w, and replace
            grad = Variable(grad_data)
            g = meta_learner(grad)  #TODO: use either h or c
            w -= F.reshape(g, shape)
            model_params[name] = w

        # Forward primary taks for training meta-leaners
        #TODO: use the same labeled data?
        y_pred = self.model(x_l0, self.model_params)
        self.loss_ml += F.softmax_cross_entropy(y_pred, y_l)

    def _train_meta_learners():
        self.cleargrads()
        self.loss_ml.backward()
        for opt in self.ml_optimizers:
            opt.update()
        self.loss_ml.unchain_backward()
        self.loss_ml = 0
            
    def test(self, x, y):
        y_pred = self.model(x, self.model_params, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

    def _cleargrads(self, ):
        for k, v in self.model_params.items():
            v.cleargrad()
        
