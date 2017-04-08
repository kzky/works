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
from recon.utils import to_device
from recon.losses import ReconstructionLoss, LSGANLoss, GANLoss, EntropyRegularizationLoss, InvariantReconstructionLoss, MeanDistanceLoss
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """Enc-MLP-Dec

    - Encoder contains linear function
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Losses
        self.recon_loss = ReconstructionLoss()
        self.er_loss = EntropyRegularizationLoss()

        # Model
        from recon.cifar10.cnn_model_000 import Encoder, MLP, Decoder
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

    def train(self, x_l, y, x_u):
        self._train(x_l, (x_l, y), y)
        self._train(x_u, (x_l, y))
        
    def _train(self, x, xy, y_0=None):
        x_, y_ = xy
        
        # Encoder/Decoder
        h = self.encoder(x)
        y_pred = self.mlp(h)

        loss = 0
        loss += self.er_loss(y_pred)   # ER loss
        if y_0 is not None:
            loss += F.softmax_cross_entropy(y_pred, y_0)  # CE loss

        x_rec = self.decoder(h)
        loss += self.recon_loss(x, x_rec) \
                + reduce(lambda u, v: u + v,
                         [self.recon_loss(u, v) \
                          for u, v in zip(self.encoder.hiddens,
                                          self.decoder.hiddens[::-1])])  # RC loss
        self.cleargrads()
        loss.backward()
        self.optimizer_enc.update()
        self.optimizer_dec.update()
        self.optimizer_mlp.update()

    def test(self, x, y):
        h = self.encoder(x, test=True)
        y_pred = self.mlp(h)
        acc = F.accuracy(y_pred, y)
        return acc

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.mlp.cleargrads()

class Experiment001(Experiment000):
    """Enc-MLP-Dec

    - Encoder contains linear function
    - ResNet
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Losses
        self.recon_loss = ReconstructionLoss()
        self.er_loss = EntropyRegularizationLoss()

        # Model
        from recon.cifar10.cnn_model_001 import Encoder, MLP, Decoder
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
        
class Experiment002(Experiment000):
    """Enc-MLP-Dec

    - Encoder contains linear function
    - ResNet
    - U-Net
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Losses
        self.recon_loss = ReconstructionLoss()
        self.er_loss = EntropyRegularizationLoss()

        # Model
        from recon.cifar10.cnn_model_002 import Encoder, MLP, Decoder
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

    def _train(self, x, xy, y_0=None):
        x_, y_ = xy
        
        # Encoder/Decoder
        h = self.encoder(x)
        y_pred = self.mlp(h)

        loss = 0
        loss += self.er_loss(y_pred)   # ER loss
        if y_0 is not None:
            loss += F.softmax_cross_entropy(y_pred, y_0)  # CE loss

        x_rec = self.decoder(h, self.encoder.hiddens)
        loss += self.recon_loss(x, x_rec)

        self.cleargrads()
        loss.backward()
        self.optimizer_enc.update()
        self.optimizer_dec.update()
        self.optimizer_mlp.update()
        
class Experiment003(Experiment000):
    """Enc-MLP-Dec

    - MeanDistanceLoss
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, n_cls=10):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.n_cls = n_cls

        # Losses
        self.md_loss = MeanDistanceLoss()
        self.er_loss = EntropyRegularizationLoss()

        # Model
        from recon.cifar10.cnn_model_000 import Encoder, MLP, Decoder
        self.encoder = Encoder(device, act)
        self.mlp = MLP(device, act)
        self.encoder.to_gpu(device) if self.device else None
        self.mlp.to_gpu(device) if self.device else None
        
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

    def train(self, x_l, y, x_u):
        self._train(x_l, (x_l, y), y)
        self._train(x_u, (x_l, y))
        
    def _train(self, x, xy, y_0=None):
        x_, y_ = xy
        
        # Encoder/Decoder
        h = self.encoder(x)
        y_pred = self.mlp(h)

        loss = 0
        loss += self.er_loss(y_pred)   # ER loss
        if y_0 is not None:
            loss += F.softmax_cross_entropy(y_pred, y_0)  # CE loss

        loss += self.md_loss(self.encoder.hiddens[-1])  # MD loss

        self.cleargrads()
        loss.backward()
        self.optimizer_enc.update()
        self.optimizer_mlp.update()
        
