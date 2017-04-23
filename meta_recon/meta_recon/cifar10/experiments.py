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
    - Gradients of auxiliary task is computed by mete-learner
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

    def setup_meta_learners(self, ):
        """
        Reconstruction loss has effects on encoder and decoder, such that
        the number of meta learners is the number of parameters of encoder since
        gradient flow on decoder is just from reconstruction loss, but on encder,
        gradient flow is from cross-entropy loss and reconstruction loss.
        """
        self.meta_enc_learners = []
        self.opt_meta_enc_learners = []
        self.last_enc_params = OrderedDict([x for x in self.encoder.namedparams()])

        #TODO: multiple layers
        # Meta-learner for encoder
        for _ in self.encoder.params():
            # meta-learner taking gradient in batch dimension
            l = L.LSTM(1, 1)
            l.to_gpu(self.device) if self.device else None
            self.meta_enc_learners.append(l)

            # optimizer of meta-learner
            opt = optimizers.Adam()
            opt.setup(l)
            opt.use_cleargrads()
            self.opt_meta_enc_learners.append(opt)

    def train(self, x_l, y, x_u):
        # Train meta learner
        self.t += 1
        if self.t > self.T:
            self.update_meta_learners(x_l, y)
            self.t = 0
            return

        # Train learner
        self._train(x_l, y)
        self._train(x_u, y=None)
        
    def _train(self, x, y=None):
        # Classifier
        h = self.encoder(x, self.last_enc_params)
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
        self.optimizer_dec.update()

    def update_parameter_by_meta_learner(self, recon_loss):
        # Reset last params
        if self.t > self.T:
            self.last_enc_params = OrderedDict()

        # Meta-learner forward for encoder
        namedparams = OrderedDict([x for x in self.encoder.namedparams()])
        for i, elm in enumerate(namedparams):  # parameter-loop
            #TODO: add loss value and affine to align dimension of the gradients
            k, p = elm
            shape = p.shape
            xp = cuda.get_array_module(p.data)
            input_ = Variable(xp.expand_dims(p.data.reshape(np.prod(shape)), axis=1))
            meta_learner = self.meta_enc_learners[i]
            g_t = meta_learner(input_)  
            p.data -= g_t.data.reshape(shape)

            # Set parameter as variable to be backward
            if self.t > self.T:
                w = p - F.reshape(g_t, shape)
                self.last_enc_params[k] = w
                                
    def update_meta_learners(self, x, y):
        # forward once for learner to chain meta-learner and learner
        h = self.encoder(x, self.last_enc_params)
        y_pred = self.mlp(h)
        loss = F.softmax_cross_entropy(y_pred, y)  # CE loss

        # backward
        self.cleargrads()
        self.cleargrads_meta_learners()
        loss.backward()

        # unchain backward
        loss.unchain_backward()

        # update
        self._update_meta_learners()

    def test(self, x, y):
        h = self.encoder(x, self.last_enc_params, test=True)
        y_pred = self.mlp(h)
        acc = F.accuracy(y_pred, y)
        return acc

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.mlp.cleargrads()

    def cleargrads_meta_learners(self, ):
        for ml in self.meta_enc_learners:
            ml.cleargrads()
        
    def _update_meta_learners(self, ):
        for opt in self.opt_meta_enc_learners:
            opt.update()

        
