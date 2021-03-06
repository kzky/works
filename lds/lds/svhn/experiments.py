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
from lds.utils import to_device
from lds.chainer_fix import BatchNormalization
from lds.losses import ReconstructionLoss, NegativeEntropyLoss, JensenShannonDivergenceLoss, KLLoss, EntropyLossForAll, EntropyLossForEachMap
from sklearn.metrics import confusion_matrix
from lds.svhn.cnn_model_000 import AutoEncoder

class Experiment000(object):
    """Regularize hiddnes of decoders with LDS.

    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution.

    Same as Experiment025 of MNIST
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = 1e-3
        self.lambda_ = 1.0
        
        # Losses
        self.recon_loss = ReconstructionLoss()
        self.ne_loss = NegativeEntropyLoss()
        
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y = self.ae.encoder(x_l)
        x_rec = self.ae.decoder(y)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(h) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(h) for h in self.ae.decoder.hiddens])
        l_ne_l = self.lambda_ * l_ne_l

        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_l = self.lambda_ * l_rec_l
        
        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l

        # Unlabeled samples
        y = self.ae.encoder(x_u)
        x_rec = self.ae.decoder(y)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(h) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(h) for h in self.ae.decoder.hiddens])
        l_ne_u = self.lambda_ * l_ne_u

        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_u = self.lambda_ * l_rec_u
        
        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
        
    def test(self, x_l, y_l):
        y = self.ae.encoder(x_l, test=True)
        acc = F.accuracy(y, y_l)
        return acc


class Experiment001(Experiment000):
    """Regularize with reconstruction and with Entropy Regularization on at the last. Same as mnist.experiment033

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment001, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.svhn.cnn_model_001 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

    def test(self, x_l, y_l):
        h = self.ae.encoder(x_l, test=True)
        y = self.ae.mlp(h, test=True)
        acc = F.accuracy(y, y_l)
        return acc

    def train(self, x_l, y_l, x_u):
        # Labeled samples
        h = self.ae.encoder(x_l)
        y = self.ae.mlp(h,)
        x_rec = self.ae.decoder(h)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y)
        l_ne_l = self.lambda_ * l_ne_l

        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_l = self.lambda_ * l_rec_l

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l

        # Unlabeled samples
        h = self.ae.encoder(x_u)
        y = self.ae.mlp(h)
        x_rec = self.ae.decoder(h)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y)
        l_ne_u = self.lambda_ * l_ne_u

        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_u = self.lambda_ * l_rec_u

        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()


class Experiment002(Experiment001):
    """Regularize with reconstruction and with Entropy Regularization on at the last. Same as mnist.experiment033

    Regularize all hiddens of MLP with LDS loss
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment002, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.svhn.cnn_model_003 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

    def test(self, x_l, y_l):
        h = self.ae.encoder(x_l, test=True)
        y = self.ae.mlp(h, test=True)
        acc = F.accuracy(y, y_l)
        return acc

    def train(self, x_l, y_l, x_u):
        # Labeled samples
        h = self.ae.encoder(x_l)
        y = self.ae.mlp(h,)
        x_rec = self.ae.decoder(h)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(h) for h in self.ae.mlp.hiddens])
        l_ne_l = self.lambda_ * l_ne_l

        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_l = self.lambda_ * l_rec_l

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l

        # Unlabeled samples
        h = self.ae.encoder(x_u)
        y = self.ae.mlp(h)
        x_rec = self.ae.decoder(h)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(h) for h in self.ae.mlp.hiddens])
        l_ne_u = self.lambda_ * l_ne_u

        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_u = self.lambda_ * l_rec_u

        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()


class Experiment003(Experiment000):
    """Regularize with reconstruction and with Entropy Regularization on at the last. Same as mnist.experiment033

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment003, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.svhn.cnn_model_003 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

    def test(self, x_l, y_l):
        h = self.ae.encoder(x_l, test=True)
        y = self.ae.mlp(h, test=True)
        acc = F.accuracy(y, y_l)
        return acc

    def train(self, x_l, y_l, x_u):
        # Labeled samples
        h = self.ae.encoder(x_l)
        y = self.ae.mlp(h,)
        x_rec = self.ae.decoder(h)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y)
        l_ne_l = self.lambda_ * l_ne_l

        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_l = self.lambda_ * l_rec_l

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l

        # Unlabeled samples
        h = self.ae.encoder(x_u)
        y = self.ae.mlp(h)
        x_rec = self.ae.decoder(h)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y)
        l_ne_u = self.lambda_ * l_ne_u

        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        l_rec_u = self.lambda_ * l_rec_u

        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
