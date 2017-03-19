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
from sslgen2.utils import to_device
from sslgen2.chainer_fix import BatchNormalization
from sslgen2.losses import ReconstructionLoss, LSGANLoss
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """Enc-Dec, Enc-Gen-Enc, Enc-Gen-Dis.

    Feature matching is taken between convolution ouputs.
    """
    def __init__(self, device=None, learning_rate=1e-3, act=F.relu, dim=100):
        # Settings
        self.device = device
        self.act = act
        self.learning_rate = learning_rate
        self.dim = dim

        # Losses
        self.recon_loss = ReconstructionLoss()
        self.lsgan_loss = LSGANLoss()

        # Model
        from sslgen2.mnist.cnn_model_000 \
            import Encoder, Decoder, Generator, Discriminator
        self.encoder = Encoder(device, act)
        self.decoder = Decoder(device, act)
        self.generator = Generator(device, act ,dim)
        self.discriminator = Discriminator(device, act ,dim)

        self.ecndoer.to_gpu(device) if self.device else None
        self.decoder.to_gpu(device) if self.device else None
        self.genrator.to_gpu(device) if self.device else None
        self.discriminator.to_gpu(device) if self.device else None
        
        # Optimizer
        self.optimizer_enc = optimizers.Adam(learning_rate)
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_dec = optimizers.Adam(learning_rate)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dec.use_cleargrads()
        self.optimizer_gen = optimizers.Adam(learning_rate)
        self.optimizer_gen.setup(self.generator)
        self.optimizer_gen.use_cleargrads()
        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_dis.use_cleargrads()

    def train(self, x):
        # Encoder/Decoder
        h = self.encoder(x)
        x_rec = self.decocer(h)
        l_rec = self.recon_loss(x, x_rec)
        self.cleargrads()
        l_rec.backward()
        self.optimizer_enc.update()
        self.optimizer_dec.update()

        # Discriminator
        h = Variable(h.data)  # disconnect
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_cpu(xp.random.rand(x.shape[0], self.dim), self.device))
        x_gen = self.generator(h, z)
        d_x_gen = self.discriminator(x_gen)
        d_x_real = self.discriminator(x)
        l_dis = self.lsgan_loss(d_x_gen, d_x_real)
        self.cleargrads()
        l_dis.backward()
        self.optimizer_dis.update()
        
        # Generator
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_cpu(xp.random.rand(x.shape[0], self.dim), self.device))
        x_gen = self.generator(h, z)
        d_x_gen = self.discriminator(x_gen)
        h_gen = self.encoder(x_gen)
        l_gen = self.lsgan_loss(d_x_gen) + self.recon_loss(h, h_gen)
        self.cleargrads()
        self.optimizer_gen.update()

    def test(self, x_l, y_l, ):
        pass

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.generator.cleargrads()
        self.discriminator.cleargrads()
