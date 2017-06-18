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

class OneByOneConvLeanrer(Chain):
    def __init__(self, ):
        maps = 16
        super(OneByOneConvLeanrer, self).__init__(
            conv0=L.ConvolutionND(ndim=1, 
                                  in_channels=3, 
                                  out_channels=maps, 
                                  ksize=1, nobias=True),
            conv1=L.ConvolutionND(ndim=1, 
                                  in_channels=maps, 
                                  out_channels=1, 
                                  ksize=1, nobias=True),
        )

    def __call__(self, x):
        # (b, m, #params)
        h = self.conv0(x)  # (1, 3, #params) -> (1, maps, #params)
        h = self.conv1(x)  # (1, maps, #params) -> (1, 1, #params)
        return h

class MetaLearner(Chain):
    def __init__(self, ):
        super(MetaLearner, self).__init__(
            ml0=OneByOneConvLeanrer(),
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
        from meta_st.cifar10.cnn_model_001_small import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None
        self.model_params = OrderedDict([x for x in self.model.namedparams()])
        
        # Optimizer, or Meta-Learner (ML)
        self.setup_meta_learners()
        
        # Initialize Meat-learners input as zero
        self.zerograds()
        
    def setup_meta_learners(self, ):
        self.meta_learners = []
        self.opt_meta_learners = []

        # Meta-learner
        for k, v in self.model_params.items():
            # meta-learner taking gradient in batch dimension
            ml = MetaLearner()
            ml.to_gpu(self.device) if self.device is not None else None
            self.meta_learners.append(ml)

            # optimizer of meta-learner
            opt = optimizers.Adam(1e-3)
            opt.setup(ml)
            opt.use_cleargrads()
            self.opt_meta_learners.append(opt)
                
    def train(self, x_l0, x_l1, y_l, x_u0, x_u1):
        # Supervised loss
        ## Forward of CE loss
        self.forward_meta_learners()
        y_pred0 = self.model(x_l0, self.model_params)
        loss_ce = F.softmax_cross_entropy(y_pred0, y_l)

        ## Cleargrads for ML
        self.cleargrad_meta_learners()

        ## Backward of CE loss
        loss_ce.backward(retain_grad=True)
        loss_ce.unchain_backward()

        ## Update ML
        self.update_meta_learners()

        # Semi-supervised loss

        ## Forward of SR loss
        self.forward_meta_learners()

        y_pred0 = self.model(x_u0, self.model_params)
        y_pred1 = self.model(x_u1, self.model_params)
        loss_rec = self.recon_loss(F.softmax(y_pred0),  F.softmax(y_pred1))

        ## Cleargrads for ML
        self.cleargrad_meta_learners()

        ## Backward of SR loss
        loss_rec.backward(retain_grad=True)
        loss_rec.unchain_backward()

        ## Update ML
        self.update_meta_learners()

    def forward_meta_learners(self, ):
        # Forward of meta-learner, i.e., parameter update
        for i, name_param in enumerate(self.model_params.items()):
            k, p = name_param
            with cuda.get_device_from_id(self.device):
                shape = p.shape
                xp = cuda.get_array_module(p.data)

                x = p.grad
                grad = xp.reshape(x, (1, 1, np.prod(shape)))
                grad_sign = xp.sign(grad)
                grad_abs = xp.abs_(grad)
                grad_separated = xp.concatenate(
                    (grad, grad_sign, grad_abs), 
                    axis=1)
                meta_learner = self.meta_learners[i]
                g = meta_learner(Variable(grad_separated))  # forward
                w = p - F.reshape(g, shape)
                self.model_params[k] = w  # parameter update

    def cleargrad_meta_learners(self, ):
        for meta_learner in self.meta_learners:
            meta_learner.cleargrads()

    def update_meta_learners(self, ):
        for opt in self.opt_meta_learners:
            opt.update()
            
    def test(self, x, y):
        y_pred = self.model(x, self.model_params, test=True)
        acc = F.accuracy(y_pred, y)
        return acc

    def zerograds(self, ):
        """For initialization of Meta-learner forward
        """
        for k, v in self.model_params.items():
            v.zerograd()  # creates the gradient region for W
        
            
