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
from lds.losses import ReconstructionLoss, NegativeEntropyLoss, JensenShannonDivergenceLoss, KLLoss, EntropyLossForAll, EntropyLossForEachMap, FrobeniousConvLoss
from sklearn.metrics import confusion_matrix
from lds.cifar10.cnn_model_000 import AutoEncoder

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
    """Regularize output with LDS loss

    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution.

    Same as Experiment031 of MNIST
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment001, self).__init__(
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
        y = self.ae.encoder(x_u)
        x_rec = self.ae.decoder(y)

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
    """Regularize output with LDS loss

    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution.

    Same as Experiment031 of MNIST
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment002, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act,
            lr_decay=lr_decay,
        )

        from lds.cifar10.cnn_model_001 import AutoEncoder

        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

class Experiment003(Experiment000):
    """Regularize hiddnes of decoders with LDS.

    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution.

    Same as Experiment031 of MNIST
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment003, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act,
            lr_decay=lr_decay,
        )

        from lds.cifar10.cnn_model_001 import AutoEncoder

        self.ae = AutoEncoder(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()
        
class Experiment004(Experiment000):
    """Regularize with reconstruction and with Entropy Regularization on at the last using model 002 (many linear)

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment004, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.cifar10.cnn_model_002 import AutoEncoderWithMLP
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
        
class Experiment005(Experiment004):
    """Regularize with reconstruction and with Entropy Regularization on at the last using model 003 (one linear). 

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment005, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.cifar10.cnn_model_003 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

class Experiment006(Experiment005):
    """Regularize with reconstruction between all hiddens except for one after 
    max_pooling and with Entropy Regularization on at the last using cnn model 
    003 (one linear). 

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment006, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.cifar10.cnn_model_004 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

class Experiment007(Experiment006):
    """Regularize with reconstruction between all hiddens except for one after 
    max_pooling and with Entropy Regularization on at the last using model 003 
    (one linear).  

    Regularize all hiddens with LDS loss

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment007, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.cifar10.cnn_model_004 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0
        
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
        h = self.ae.encoder(x_u)
        y = self.ae.mlp(h)
        x_rec = self.ae.decoder(h)

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

class Experiment008(Experiment007):
    """Regularize with reconstruction between all hiddens except for one after 
    max_pooling and with Entropy Regularization on at the last using model 003 
    (one linear).  

    Regularize all hiddens of MLP with LDS loss

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment008, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.cifar10.cnn_model_005 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0
        
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
        
class Experiment009(Experiment006):
    """Regularize with reconstruction between all hiddens except for one after 
    max_pooling and with Entropy Regularization on at the last using cnn model 
    003 (one linear) and using Frobineus Conv Loss.

    NOTE: Frobineus Conv Loss take too long time, so GIVE UP to use this.

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment009, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        # Frobenious Conv Loss
        self.fc_loss = FrobeniousConvLoss(self.device)
        
        # Model
        from lds.cifar10.cnn_model_004 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

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

        # frobenious conv loss
        l_fc_l = reduce(lambda x, y: x + y, 
                        [self.fc_loss(h) for h in self.ae.encoder.hiddens]) \
                        + reduce(lambda x, y: x + y, 
                                 [self.fc_loss(h) for h in self.ae.decoder.hiddens])
        l_fc_l = self.lambda_ * l_fc_l

        # loss for labeled samples
        loss_l = l_ce_l + l_ne_l + l_rec_l + l_fc_l

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

        # frobenious conv loss
        l_fc_u = reduce(lambda x, y: x + y, 
                        [self.fc_loss(h) for h in self.ae.encoder.hiddens]) \
                        + reduce(lambda x, y: x + y, 
                                 [self.fc_loss(h) for h in self.ae.decoder.hiddens])
        l_fc_u = self.lambda_ * l_fc_u

        # loss for unlabeled samples
        loss_u = l_ne_u + l_rec_u + l_fc_u

        loss = loss_l + loss_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()

class Experiment010(Experiment006):
    """Regularize with reconstruction between all hiddens except for one after 
    max_pooling and with Entropy Regularization on at the last using cnn model 
    003 (one linear).

    Train labeled samples and unlabeled samples separately.
    When training all labeled samples, use one batch of unlabeled samples.

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment010, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        # Frobenious Conv Loss
        self.fc_loss = FrobeniousConvLoss(self.device)
        
        # Model
        from lds.cifar10.cnn_model_004 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

    def train_with_labeled(self, x_l, y_l, ):
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

        # Backward and Update
        self.ae.cleargrads()
        loss_l.backward()
        self.optimizer.update()

    def train_with_unlabeled(self, x_u):
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

        # Backward and Update
        self.ae.cleargrads()
        loss_u.backward()
        self.optimizer.update()
        
class Experiment011(Experiment006):
    """
    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution and with ResNet
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment011, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        from lds.cifar10.cnn_model_005 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

class Experiment012(Experiment006):
    """
    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution and with ResNet 3
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment012, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        from lds.cifar10.cnn_model_006 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0
        
class Experiment013(Experiment006):
    """
    Using max pooling in Encoder and deconvolution instead of unpooling in 
    Decoder, and regularize NOT between maxpooing and upsample 
    deconvolution and with ResNet 3 
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment013, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        from lds.cifar10.cnn_model_007 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0
        

class Experiment014(Experiment005):
    """Regularize with reconstruction between all hiddens except for one after 
    max_pooling and with Entropy Regularization on at the last using cnn model 
    003 (one linear). 

    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, lr_decay=False):
        super(Experiment014, self).__init__(
            device=device,
            learning_rate=learning_rate,
            act=act, 
        )
        
        # Model
        from lds.cifar10.cnn_model_004 import AutoEncoderWithMLP
        self.ae = AutoEncoderWithMLP(act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()

        self.lambda_ = 1.0

    def train(self, x_l, y_l, x_u):
        # Labeled and unlabeled samples
        loss_l = self._compute_l_loss(x_l, y_l, x_u)
        #loss_u = self._compute_u_loss(x_l, y_l, x_u)
        loss = loss_l #+ loss_u

        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()

        # Noisy samples
        loss_l = self._compute_loss_with_noise(x_l, y_l, x_u)
        #loss_u = self._compute_loss_with_noise(x_u, y_l, x_u)
        loss = loss_l #+ loss_u

        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()

    def _compute_loss_with_noise(self, x_l, y_l, x_u):
        x_l_0 = x_l
        x_l_grad = Variable(x_l.grad)
        shape = x_l_grad.shape
        bs = shape[0]
        d = np.prod(shape[1:])
        noise = F.reshape(F.normalize(F.reshape(x_l_grad, (bs, d))), shape)
        x_l_noise = x_l + noise * 0.1

        # label confidence loss
        h = self.ae.encoder(x_l)
        y = self.ae.mlp(h,)

        h = self.ae.encoder(x_l_0)
        y_0 = self.ae.mlp(h,)
        l_lc_l = 0
        l_lc_l += F.mean_squared_error(y_0, y)

        loss = l_lc_l * self.lambda_
        return loss
    
    def _compute_l_loss(self, x_l, y_l, x_u, label_only=False):
        # Labeled samples
        h = self.ae.encoder(x_l)
        y = self.ae.mlp(h,)
        x_rec = self.ae.decoder(h)

        # cronss entropy loss
        l_ce_l = 0
        l_ce_l += F.softmax_cross_entropy(y, y_l)

        ## negative entropy loss
        #l_ne_l = 0
        #l_ne_l += self.ne_loss(y)
        #l_ne_l = self.lambda_ * l_ne_l

        ## reconstruction loss
        #l_rec_l = 0
        #l_rec_l += self.recon_loss(x_l, x_rec) \
        #           + reduce(lambda x, y: x + y,
        #                    [self.recon_loss(x, y) for x, y in zip(
        #                        self.ae.encoder.hiddens,
        #                        self.ae.decoder.hiddens[::-1])])
        #l_rec_l = self.lambda_ * l_rec_l

        # loss for labeled samples
        if label_only:
            return l_ce_l

        loss_l = l_ce_l #+ l_ne_l + l_rec_l
        return loss_l

    def _compute_u_loss(self, x_l, y_l, x_u):
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
        return loss_u
        
