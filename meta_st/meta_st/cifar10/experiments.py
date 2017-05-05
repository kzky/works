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

class MetaLearner(Chain):
    def __init__(self, inmap=4, midmap=4, outmap=1, ):
        super(MetaLearner, self).__init__(
            l0=L.LSTM(inmap, midmap, 
                      forget_bias_init=1e12, 
                       lateral_init=1e-12*np.random.randn(1, 1), 
                       upward_init=1e-12*np.random.randn(1, 1)),
            l1=L.LSTM(midmap, outmap, 
                      forget_bias_init=1e12, 
                       lateral_init=1e-12*np.random.randn(1, 1), 
                      upward_init=1e-12*np.random.randn(1, 1)),
        )

    def __call__(self, h):
        h = self.l0(h)
        h = self.l1(h)
        #h = F.tanh(h) * 1e-6
        return h
        

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
        self.setup_meta_learners()

    def setup_meta_learners(self, ):
        #TODO: multiple layers, modification of inputs
        self.meta_learners = []
        self.opt_meta_learners = []

        # Meta-learner
        for _ in self.model_params:
            # meta-learner taking gradient in batch dimension
            ml = MetaLearner(4, 4, 1)
            ml.to_gpu(self.device) if self.device else None
            self.meta_learners.append(ml)

            # optimizer of meta-learner
            opt = optimizers.Adam(1e-3)
            opt.setup(ml)
            opt.use_cleargrads()
            self.opt_meta_learners.append(opt)

    def update_parameter_by_meta_learner(self, model_params, loss):
        namedparams = model_params
        for i, elm in enumerate(namedparams.items()):  # parameter-loop
            k, p = elm
            with cuda.get_device(self.device):
                shape = p.shape
                xp = cuda.get_array_module(p.data)
                # normalize grad
                x = p.grad
                p_val = 10
                grad0 = xp.where(xp.absolute(x) > xp.exp(-p_val), 
                                   xp.log(xp.absolute(x))/p_val, -1)
                grad1 = xp.where(xp.absolute(x) > xp.exp(-p_val), 
                                   xp.sign(x), xp.exp(p_val)*x)
                grad0 = xp.reshape(grad0, (np.prod(shape), ))
                grad1 = xp.reshape(grad1, (np.prod(shape), ))
                grad0 = xp.expand_dims(grad0, axis=1)
                grad1 = xp.expand_dims(grad1, axis=1)
                input_grad = xp.concatenate((grad0, grad1), axis=1)

                # normalize loss
                x = loss.data
                loss0 = xp.where(xp.sign(x) > xp.exp(-p_val), 
                                   xp.log(xp.sign(x))/p_val, -1)
                loss1 = xp.where(xp.sign(x) > xp.exp(-p_val), 
                                   xp.sign(x), xp.exp(p_val)*x)
                loss0 = xp.expand_dims(loss0, axis=0)
                loss1 = xp.expand_dims(loss1, axis=0)
                input_loss = xp.concatenate((loss0, loss1))
                input_loss = xp.broadcast_to(input_loss, 
                                             (input_grad.shape[0], 2))
                # input
                input_ = xp.concatenate((input_grad, input_loss), axis=1)
                meta_learner = self.meta_learners[i]
                g = meta_learner(Variable(input_.astype(xp.float32))) # forward of meta-learner
                #g = g * 1e-12
                p.data -= g.data.reshape(shape)

            # Set parameter as variable to be backproped
            if self.t  == self.T:
                w = p - F.reshape(g, shape)
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

        self.cleargrads()
        for meta_learner in self.meta_learners:
            meta_learner.cleargrads()
        loss_ce.backward()

        for opt in self.opt_meta_learners:
            opt.clip_grads(0.1)
            opt.update()

        loss_ce.unchain_backward()  #TODO: here is a proper place to unchain?

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
        self.update_parameter_by_meta_learner(self.model_params, loss_rec)
                        
    def test(self, x, y):
        y_pred = self.model(x, self.model_params, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

    def cleargrads(self, ):
        for k, v in self.model_params.items():
            v.cleargrad()
        
class Experiment001(object):
    """
    - Stochastic Regularization
    - ConvPool-CNN-C (Springenberg et al., 2014, Salimans&Kingma (2016))
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
        
