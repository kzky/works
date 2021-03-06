"""Experiments

Conv meta-learner for W

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
        in_channels = 1
        maps = 16
        super(OneByOneConvLeanrer, self).__init__(
            conv0=L.ConvolutionND(ndim=1, 
                                  in_channels=in_channels, 
                                  out_channels=maps, 
                                  ksize=1, initial_bias=None),
            conv1=L.ConvolutionND(ndim=1, 
                                  in_channels=maps, 
                                  out_channels=1, 
                                  ksize=1, initial_bias=None),
        )
        self.w_accum = None

    def __call__(self, x):
        if self.w_accum is None:
            self.w_accum = x
            return self.w_accum

        h = self.conv0(x)
        h = self.conv1(h)
        h = F.sigmoid(h)
        self.w_accum = h * self.w_accum + (1 - h) * x
        return self.w_accum

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

        # Loss
        self.recon_loss = ReconstructionLoss()

        # Model
        from meta_st.cifar10.cnn_model_001_small import Model
        self.model = Model(device, act)
        self.model.to_gpu(device) if device is not None else None
        self.model_params = OrderedDict([x for x in self.model.namedparams()])

        # Optimizer for model
        self.optimizer = Adam()
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
        
        # Optimizer, or Meta-Learner (ML)
        self.setup_meta_learners()
        
        # Initialize Meat-learners input as zero
        #self.zerograds()
        
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
        self.forward_meta_learners()  #TODO: init ML'W as 0?
        y_pred0 = self.model(x_l0, self.model_params)
        loss_ce = F.softmax_cross_entropy(y_pred0, y_l)

        ## Backward of CE loss
        self.cleargrad_meta_learners()
        self.cleargrads()
        loss_ce.backward(retain_grad=True)
        loss_ce.unchain_backward()
        
        ## Optimizer update
        self.optimizer.update(self.model_params)
        self.update_meta_learners()

        # Semi-supervised loss
        self.forward_meta_learners()
        y_pred0 = self.model(x_u0, self.model_params)
        y_pred1 = self.model(x_u1, self.model_params)
        loss_rec = self.recon_loss(F.softmax(y_pred0),  F.softmax(y_pred1))

        ## Backward of SR loss
        self.cleargrad_meta_learners()
        self.cleargrads()
        loss_rec.backward(retain_grad=True)
        loss_rec.unchain_backward()

        ## Optimizer update
        self.optimizer.update(self.model_params)
        self.update_meta_learners()

    def forward_meta_learners(self, ):
        # Forward of meta-learner, i.e., parameter update
        for i, name_param in enumerate(self.model_params.items()):
            k, p = name_param
            with cuda.get_device_from_id(self.device):
                shape = p.shape
                xp = cuda.get_array_module(p.data)

                w_data = p.data  # meta learner is gated-reccurent unit for W not for G
                w_data = xp.reshape(w_data, (1, 1, np.prod(shape)))
                meta_learner = self.meta_learners[i]
                w_accum = meta_learner(Variable(w_data))  # forward
                w_accum = F.reshape(w_accum, shape)
                self.model_params[k] = w_accum

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

    def cleargrads(self, ):
        """For initialization of Meta-learner forward
        """
        for k, v in self.model_params.items():
            v.cleargrad()  # creates the gradient region for W
        
    def zerograds(self, ):
        """For initialization of Meta-learner forward
        """
        for k, v in self.model_params.items():
            v.zerograd()  # creates the gradient region for W
