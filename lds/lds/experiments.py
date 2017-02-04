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
from losses import ReconstructionLoss, NegativeEntropyLoss, GANLoss, JensenShannonDivergenceLoss
from sklearn.metrics import confusion_matrix
from lds.cnn_model import AutoEncoder

class Experiment(object):

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = 1e-3

        # Model
        self.ae = AutoEncoder(act=act)
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

class Experiment001(Experiment):
    """Decoder predict label and use NE loss on these predictions.
    """

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu):
        super(Experiment001, self).__init__(
            device=device, learning_rate=learning_rate, act=act
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

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu):
        super(Experiment003, self).__init__(
            device=device, learning_rate=learning_rate, act=act
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
        loss_u = l_ne_u + l_rec_u + l_ljd_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
    
