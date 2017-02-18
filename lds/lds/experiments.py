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
from utils import to_device
from chainer_fix import BatchNormalization
from losses import ReconstructionLoss, NegativeEntropyLoss, JensenShannonDivergenceLoss
from sklearn.metrics import confusion_matrix
from lds.cnn_model import AutoEncoder

class Experiment(object):

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = 1e-3
        self.lr_decay = lr_decay

        # Model
        self.ae = AutoEncoder(act=act, lr_decay=lr_decay)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()
        
        # Losses
        self.recon_loss = ReconstructionLoss()
        self.ne_loss = NegativeEntropyLoss()
        
    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y = self.ae.encoder(x_l)
        x_rec = self.ae.decoder(y)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l) \
                  + reduce(lambda x, y: x + y, 
                           [F.softmax_cross_entropy(y_, y_l) for y_ in self.ae.encoder.classifiers])

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers])
        
        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l

        # Unlabeled samples
        y = self.ae.encoder(x_u)
        x_rec = self.ae.decoder(y)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers])
        
        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
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
        
        accs = [F.accuracy(y_, y_l) \
                for y_ in self.ae.encoder.classifiers] + [acc] 
        
        return accs

class Experiment000(Experiment):
    """Regularize hiddnes of decoders with LDS.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment000, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            lr_decay=lr_decay,
        )
        
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
        
        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

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
        
        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
    
class Experiment001(Experiment):
    """Decoder predicts labels and use NE loss on these predictions.
    """

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment001, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )
        
        # Model
        from lds.cnn_model_001 import AutoEncoder
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
        l_ce_l += F.softmax_cross_entropy(y, y_l) \
                  + reduce(lambda x, y: x + y, 
                           [F.softmax_cross_entropy(y_, y_l) \
                            for y_ in self.ae.encoder.classifiers]) \
                                + reduce(lambda x, y: x + y, 
                                         [F.softmax_cross_entropy(y_, y_l) \
                                          for y_ in self.ae.decoder.classifiers])

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in self.ae.decoder.classifiers])
        
        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l

        # Unlabeled samples
        y = self.ae.encoder(x_u)
        x_rec = self.ae.decoder(y)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in self.ae.decoder.classifiers])
        
        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])
        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()

    def test(self, x_l, y_l):
        y = self.ae.encoder(x_l, test=True)
        x = self.ae.decoder(y, test=True)
        
        acc = F.accuracy(y, y_l)
        
        accs = [F.accuracy(y_, y_l) \
                for y_ in self.ae.encoder.classifiers] \
                    + [F.accuracy(y_, y_l)\
                       for y_ in self.ae.decoder.classifiers] \
                           + [acc] 
        
        return accs
    
class Experiment002(Experiment001):
    """Decoder predict label.
    It uses NE loss on these predictions and L2 loss between predictions between
    encoder and decoder.
    """
        
    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y = self.ae.encoder(x_l)
        x_rec = self.ae.decoder(y)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l) \
                  + reduce(lambda x, y: x + y, 
                           [F.softmax_cross_entropy(y_, y_l) \
                            for y_ in self.ae.encoder.classifiers]) \
                                + reduce(lambda x, y: x + y, 
                                         [F.softmax_cross_entropy(y_, y_l) \
                                          for y_ in self.ae.decoder.classifiers])
    
        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in self.ae.decoder.classifiers])
        
        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # label reconstruction loss
        l_lrec_l = reduce(lambda x, y: x + y,
                          [self.recon_loss(x, y) for x, y in zip(
                              self.ae.encoder.classifiers,
                              self.ae.decoder.classifiers[::-1])])

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l + l_lrec_l

        # Unlabeled samples
        y = self.ae.encoder(x_u)
        x_rec = self.ae.decoder(y)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in self.ae.decoder.classifiers])
        
        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # label reconstruction loss
        l_lrec_u = reduce(lambda x, y: x + y,
                          [self.recon_loss(x, y) for x, y in zip(
                              self.ae.encoder.classifiers,
                              self.ae.decoder.classifiers[::-1])])
        
        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u + l_lrec_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
    
class Experiment003(Experiment002):
    """Decoder predict label.
    It uses NE loss on these predictions and Jensen Shannon Divergence 
    between predictions between encoder and decoder.
    """

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment003, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )

        self.jsd_loss = JensenShannonDivergenceLoss()
        
    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y = self.ae.encoder(x_l)
        x_rec = self.ae.decoder(y)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l) \
                  + reduce(lambda x, y: x + y, 
                           [F.softmax_cross_entropy(y_, y_l) \
                            for y_ in self.ae.encoder.classifiers]) \
                                + reduce(lambda x, y: x + y, 
                                         [F.softmax_cross_entropy(y_, y_l) \
                                          for y_ in self.ae.decoder.classifiers])

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in self.ae.decoder.classifiers])
        
        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # label jsd loss
        l_ljsd_l = reduce(lambda x, y: x + y,
                          [self.jsd_loss(x, y) for x, y in zip(
                              self.ae.encoder.classifiers,
                              self.ae.decoder.classifiers[::-1])])

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l + l_ljsd_l

        # Unlabeled samples
        y = self.ae.encoder(x_u)
        x_rec = self.ae.decoder(y)

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in self.ae.encoder.classifiers]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in self.ae.decoder.classifiers])
        
        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                self.ae.encoder.hiddens,
                                self.ae.decoder.hiddens[::-1])])

        # unlabel jsd loss
        l_ljsd_u = reduce(lambda x, y: x + y,
                          [self.jsd_loss(x, y) for x, y in zip(
                              self.ae.encoder.classifiers,
                              self.ae.decoder.classifiers[::-1])])
        
        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u + l_ljsd_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
    
class Experiment004(Experiment000):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment004, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )
        
        # Model
        from lds.cnn_model_002 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment005(Experiment):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment005, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment006(Experiment000):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment006, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment007(Experiment001):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment007, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment008(Experiment002):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment008, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment009(Experiment):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment009, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment010(Experiment000):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment010, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment011(Experiment001):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment011, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment012(Experiment002):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment012, self).__init__(
            device=device, learning_rate=learning_rate, act=act, lr_decay=lr_decay
        )        
        # Model
        from lds.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

