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
from meta_recon.utils import to_device
from meta_recon.losses import ReconstructionLoss, LSGANLoss, GANLoss, EntropyRegularizationLoss, InvariantReconstructionLoss, MeanDistanceLoss, DistanceLoss
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """Enc-MLP-Dec

    - Encoder contains linear function
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10, T=5):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls
        self.T = T
        self.t = 0

        # Losses
        self.meta_recon_loss = ReconstructionLoss()
        self.er_loss = EntropyRegularizationLoss()

        # Model
        from meta_recon.cifar10.cnn_model_000 import Encoder, MLP, Decoder
        self.encoder = Encoder(device, act)
        self.mlp = MLP(device, act)
        self.decoder = Decoder(device, act)
        self.encoder.to_gpu(device) if self.device else None
        self.mlp.to_gpu(device) if self.device else None
        self.decoder.to_gpu(device) if self.device else None
        
        # Optimizer
        self.optimizer_enc = optimizers.Adam(learning_rate)
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_mlp = optimizers.Adam(learning_rate)
        self.optimizer_mlp.setup(self.mlp)
        self.optimizer_mlp.use_cleargrads()
        self.optimizer_dec = optimizers.Adam(learning_rate)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dec.use_cleargrads()

        # Meta model and its optimizer
        self.setup_meta_learners()

    def meta_learners(self, ):
        """
        Reconstruction loss has effects on encoder and decoder, such that
        the number of meta learners is the number of parameters of encoder and
        decoder.
        """
        
        self.meta_enc_learners = []
        self.opt_meta_enc_learners = []
        self.meta_dec_learners = []
        self.opt_dec_learners = []

        #TODO: multiple layers
        # Encoder
        for _ in self.endocer.params():
            l = L.LSTM(2, 2)  # (grad, loss)
            l.to_gpu(device) if self.device else None
            self.meta_enc_learners.append(l)

            opt = optimizers.Adam()
            opt.setup(l)
            opt.use_cleargrads()
            self.opt_meta_enc_learners.append(opt)

        # Decoder
        for _ in self.decoder.params():
            l = L.LSTM(2, 2)  # (grad, loss)
            l.to_gpu(device) if self.device else None
            self.meta_dec_learners.append(l)

            opt = optimizers.Adam()
            opt.setup(l)
            opt.use_cleargrads()
            self.opt_meta_dec_learners.append(opt)
        
    def train(self, x_l, y, x_u):
        # Train meta learner
        self.t += 1
        if self.t > T:
            self.update_meta_learners(x_l, y)
            self.t = 0
            return

        # Train learner
        self._train(x_l, y)
        self._train(x_u, y=None)
        
    def _train(self, x, y=None):
        # Classifier
        h = self.encoder(x)
        y_pred = self.mlp(h)
        loss = 0
        if y is not None:
            loss += F.softmax_cross_entropy(y_pred, y)  # CE loss

        self.cleargrads()
        loss.backward()
        self.optimizer_enc.update()
        self.optimizer_mlp.update()
        
        # Encoder/Decoder
        x_rec = self.decoder(h)
        loss += self.meta_recon_loss(x, x_rec) \
                + reduce(lambda u, v: u + v,
                         [self.meta_recon_loss(u, v) \
                          for u, v in zip(self.encoder.hiddens,
                                          self.decoder.hiddens[::-1])])  # RC loss
        self.cleargrads()
        loss.backward()
        self.update_parameter_by_meta_learner(loss)

    def update_parameter_by_meta_learner(self, recon_loss):
        #TODO: Get parameters only, using slice?
        #TODO: Unchain backward
        # Encoder
        for i in range(self.encoder.params()):
            xp = cuda.get_array_module(p.data)
            shape = p.shape
            with cuda.get_device(self.device):
                g_1d = xp.expand_dims(p.grad.reshape(np.prod(shape)), axis=1)
                recon_loss_tiled = xp.zeros_like(g_1d)
                xp.copyto(recon_loss_tiled, recon_loss.data)
                input_ = Variable(xp.concatenate((g_1d, recon_loss_tiled), axis=1))
                
                # update parameter at t
                meta_learner = self.meta_enc_learners[i]
                g_t = meta_learner(input_)  
                p.data -= g_t.data.reshape(shape)
                                
        # Decoder
        for i in range(self.decoder.target.params()):
            xp = cuda.get_array_module(p.data)
            shape = p.shape
            with cuda.get_device(self.device):
                g_1d = xp.expand_dims(p.grad.reshape(np.prod(shape)), axis=1)
                recon_loss_tiled = xp.zeros_like(g_1d)
                xp.copyto(recon_loss_tiled, recon_loss.data)
                input_ = Variable(xp.concatenate((g_1d, recon_loss_tiled), axis=1))
                
                # update parameter at t
                meta_learner = self.meta_dec_learners[i]
                g_t = meta_learner(input_)  
                p.data -= g_t.data.reshape(shape)
            
    def update_meta_learners(self, x, y):
        h = self.encoder(x)
        y_pred = self.mlp(h)
        loss = 0
        loss += F.softmax_cross_entropy(y_pred, y)  # CE loss
    
    def test(self, x, y):
        h = self.encoder(x, test=True)
        y_pred = self.mlp(h)
        acc = F.accuracy(y_pred, y)
        return acc

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.mlp.cleargrads()

