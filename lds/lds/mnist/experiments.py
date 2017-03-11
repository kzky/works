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
from chainer_fix import BatchNormalization
from losses import ReconstructionLoss, NegativeEntropyLoss, JensenShannonDivergenceLoss, KLLoss, EntropyLossForAll, EntropyLossForEachMap
from sklearn.metrics import confusion_matrix
from lds.mnist.cnn_model import AutoEncoder

class Experiment(object):

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
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

class Experiment000(Experiment):
    """Regularize hiddnes of decoders with LDS.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment000, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            
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
            device=device, learning_rate=learning_rate, act=act, 
        )
        
        # Model
        from lds.mnist.cnn_model_001 import AutoEncoder
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
            device=device, learning_rate=learning_rate, act=act, 
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
            device=device, learning_rate=learning_rate, act=act, 
        )
        
        # Model
        from lds.mnist.cnn_model_002 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment005(Experiment):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment005, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment006(Experiment000):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment006, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment007(Experiment001):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment007, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment008(Experiment002):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment008, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_003 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment009(Experiment):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment009, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment010(Experiment000):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment010, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment011(Experiment001):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment011, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment012(Experiment002):
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment012, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        
        # Model
        from lds.mnist.cnn_model_004 import AutoEncoder
        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment013(Experiment008):
    
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment013, self).__init__(
            device=device, learning_rate=learning_rate, act=act, 
        )        

        self.kl_loss = KLLoss()

    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y_pred_l = self.ae.encoder(x_l)
        preds_enc_l = self.ae.encoder.classifiers
        hiddens_enc_l = self.ae.encoder.hiddens
        x_rec_l = self.ae.decoder(y_pred_l)
        preds_dec_l = self.ae.decoder.classifiers
        hiddens_dec_l = self.ae.decoder.hiddens

        # cross entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y_pred_l, y_l) \
                  + reduce(lambda x, y: x + y, 
                           [F.softmax_cross_entropy(y_, y_l) \
                            for y_ in preds_enc_l]) \
                                + reduce(lambda x, y: x + y, 
                                         [F.softmax_cross_entropy(y_, y_l) \
                                          for y_ in preds_dec_l])
    
        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self.ne_loss(y_pred_l) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in preds_enc_l]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in preds_dec_l])
        
        # reconstruction loss
        l_rec_l = 0
        l_rec_l += self.recon_loss(x_l, x_rec_l) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                hiddens_enc_l,
                                hiddens_dec_l[::-1])])

        # label reconstruction loss
        l_lrec_l = reduce(lambda x, y: x + y,
                          [self.recon_loss(x, y) for x, y in zip(
                              preds_enc_l,
                              preds_dec_l[::-1])])

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l + l_lrec_l

        # Unlabeled samples
        y_pred_u = self.ae.encoder(x_u)
        preds_enc_u = self.ae.encoder.classifiers
        hiddens_enc_u = self.ae.encoder.hiddens
        x_rec_u = self.ae.decoder(y_pred_u)
        preds_dec_u = self.ae.decoder.classifiers
        hiddens_dec_u = self.ae.decoder.hiddens

        # negative entropy loss
        l_ne_u = 0
        l_ne_u += self.ne_loss(y_pred_u) \
                  + reduce(lambda x, y: x + y, 
                           [self.ne_loss(y_) for y_ in preds_enc_u]) \
                           + reduce(lambda x, y: x + y, 
                                    [self.ne_loss(y_) for y_ in preds_dec_u])
        
        # reconstruction loss
        l_rec_u = 0
        l_rec_u += self.recon_loss(x_u, x_rec_u) \
                   + reduce(lambda x, y: x + y,
                            [self.recon_loss(x, y) for x, y in zip(
                                hiddens_enc_u,
                                hiddens_dec_u[::-1])])

        # label reconstruction loss
        l_lrec_u = reduce(lambda x, y: x + y,
                          [self.recon_loss(x, y) for x, y in zip(
                              preds_enc_u,
                              preds_dec_u[::-1])])

        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u + l_lrec_u

        # Labeled and Unlabeled samples
        l_rec_lu = - reduce(lambda x, y: x + y,
                           [self.kl_loss(x, y) for x, y in zip(
                               hiddens_enc_l,
                               hiddens_enc_u)]) - \
                               reduce(lambda x, y: x + y,
                                      [self.kl_loss(x, y) for x, y in zip(
                                          hiddens_dec_l,
                                          hiddens_dec_u)])

        loss = loss_l + loss_u + l_rec_lu

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()


class Experiment014(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    LDS for all values of the output of Convolution
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment014, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        self.ne_loss = EntropyLossForAll()
        
class Experiment015(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    LDS for each map
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment015, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        self.ne_loss = EntropyLossForEachMap()
        

class Experiment016(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    Regularize with maxpooling.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment016, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            
        )
        self.f_pool = F.max_pooling_2d
        
    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y = self.ae.encoder(x_l)
        x_rec = self.ae.decoder(y)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self._ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self._ne_loss(h) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self._ne_loss(h) for h in self.ae.decoder.hiddens])
        
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
        l_ne_u += self._ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self._ne_loss(h, ) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self._ne_loss(h) for h in self.ae.decoder.hiddens])
        
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

    def _ne_loss(self, h, ):
        """Entropy regularization depending on the output dimension
        """

        shape = h.shape

        # Linear
        if len(shape) == 2:
            h = self.ne_loss(h)
            return h
            
        # Convolution2D
        if len(shape) == 4:
            h = self.f_pool(h, (2, 2), )
            h = self.ne_loss(h)
            return h
            
class Experiment017(Experiment016):
    """Regularize hiddnes of decoders with LDS.

    Regularize with maxpooling.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment017, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            
        )
        self.f_pool = F.average_pooling_2d

class Experiment018(Experiment016):
    """Regularize hiddnes of decoders with LDS.

    Regularize with maxpooling.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment018, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            
        )
        self.f_pool = F.max_pooling_2d

    def _ne_loss(self, h, ):
        """Entropy regularization depending on the output dimension 
        using various deterministic receptive fields.
        """
        shape = h.shape

        # Linear
        if len(shape) == 2:
            h = self.ne_loss(h)
            return h

        if len(shape) == 4:
            v_list = []
            sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]
            for size in sizes:
                h_ = self.f_pool(h, size, )
                h_ = self.ne_loss(h_)
                v_list.append(h_)
            h = self.ne_loss(h)
            v_list.append(h)
            
            return reduce(lambda x, y: x+y, v_list)

class Experiment019(Experiment018):
    """Regularize hiddnes of decoders with LDS.

    Regularize with maxpooling.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment019, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            
        )
        self.f_pool = F.average_pooling_2d

class Experiment020(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    Stochastic LDS for each dimension of the output of Convolution
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment020, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
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
        l_ne_l += self._ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self._ne_loss(h) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self._ne_loss(h) for h in self.ae.decoder.hiddens])
        
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
        l_ne_u += self._ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self._ne_loss(h, ) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self._ne_loss(h) for h in self.ae.decoder.hiddens])
        
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

    def _ne_loss(self, h, ):
        if np.random.randint(2) == 0:
            return 0
        else:
            return self.ne_loss(h)
    
class Experiment021(Experiment020):
    """Regularize hiddnes of decoders with LDS.

    Stochastic LDS for all values of the output of Convolution
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment021, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )

    def _ne_loss(self, h, ):
        h = F.dropout(h)
        h = self.ne_loss(h)
        return h

class Experiment022(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    Entropy Regularization for a certain size of receptive field.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment022, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        self.ne_loss = EntropyLossForAll()
        self.size = 2
        self.lambda_ = 1.0
        
    def train(self, x_l, y_l, x_u):
        # Labeled samples
        y = self.ae.encoder(x_l)
        x_rec = self.ae.decoder(y)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        # negative entropy loss
        l_ne_l = 0
        l_ne_l += self._ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self._ne_loss(h) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self._ne_loss(h) for h in self.ae.decoder.hiddens])
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
        l_ne_u += self._ne_loss(y) \
                  + reduce(lambda x, y: x + y, 
                           [self._ne_loss(h, ) for h in self.ae.encoder.hiddens]) \
                           + reduce(lambda x, y: x + y, 
                                    [self._ne_loss(h) for h in self.ae.decoder.hiddens])
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

    def _ne_loss(self, h, ):
        shape = h.shape

        # Linear
        if len(shape) == 2:
            h = self.ne_loss(h)
            return h

        b, d, w, w = shape
        v_list = []
        s = self.size

        for i in range(0, w - s):
            h_ = h[:, :, i:i+s, i:i+s]
            h_ = self.ne_loss(h_)
            v_list.append(h_)

        return reduce(lambda x, y: x + y, v_list)
    
class Experiment023(Experiment022):
    """Regularize hiddnes of decoders with LDS.

    Entropy Regularization for multi-scale receptive field.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment023, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        self.ne_loss_all = EntropyLossForAll()
        self.ne_loss = NegativeEntropyLoss()
        self.lambda_ = 1.0
        
    def _ne_loss(self, h, ):
        shape = h.shape

        # Linear
        if len(shape) == 2:
            h = self.ne_loss(h)
            return h

        b, d, w, w = shape
        v_list = []
        sizes = [2, 3, ]

        # Multi scale
        for s in sizes:
            for i in range(0, w - s):
                h_ = h[:, :, i:i+s, i:i+s]
                h_ = self.ne_loss_all(h_)
                v_list.append(h_)
        h_ = self.ne_loss(h)
        v_list.append(h_)

        return reduce(lambda x, y: x + y, v_list)
        
class Experiment024(Experiment022):
    """Regularize hiddnes of decoders with LDS.

    Entropy Regularization for a certain size of receptive field.
    Different size from Experiment022
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment024, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )

        self.ne_loss = EntropyLossForAll()
        self.size = 3
        self.lambda_ = 1.0

class Experiment025(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    Using the lambda decay for entropy regularizatoin
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment025, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
            
        )
        self.lambda_ = 1.0
        
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
        
