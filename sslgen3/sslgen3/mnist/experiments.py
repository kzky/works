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
from sslgen3.utils import to_device
from sslgen3.losses import ReconstructionLoss, LSGANLoss
from sklearn.metrics import confusion_matrix

class Experiment000(object):
    """Enc-Dec, Enc-Gen-Enc, Enc-Gen-Dis.

    - Feature matching is taken between convolution ouputs.
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
        from sslgen3.mnist.cnn_model_000 \
            import Encoder, MLP, Decoder, Generator0, Discriminator,
        self.encoder = Encoder(device, act)
        self.mlp = MLP(device, act)
        self.decoder = Decoder(device, act)
        self.generator0 = Generator0(device, act ,dim)
        self.discriminator = Discriminator(device, act)

        self.encoder.to_gpu(device) if self.device else None
        self.mlp.to_gpu(device) if self.device else None
        self.decoder.to_gpu(device) if self.device else None
        self.generator0.to_gpu(device) if self.device else None
        self.discriminator.to_gpu(device) if self.device else None
        
        # Optimizer
        self.optimizer_enc = optimizers.Adam(learning_rate)
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_mlp = optimizers.Adam(learning_rate)
        self.optimizer_mlp.setup(self.encoder)
        self.optimizer_mlp.use_cleargrads()
        self.optimizer_dec = optimizers.Adam(learning_rate)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dec.use_cleargrads()
        self.optimizer_gen = optimizers.Adam(learning_rate)
        self.optimizer_gen.setup(self.generator0)
        self.optimizer_gen.use_cleargrads()
        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_dis.use_cleargrads()

    def train(self, x_l, y_l, x_u):
        self._train(x_l, y_l)
        self._train(x_u)

    def _train(self, x, y=None):
        # Encoder/Decoder/MLP
        h = self.encoder(x)
        x_rec = self.decoder(h)
        l_rec = self.recon_loss(x, x_rec) \
                + reduce(lambda x, y: x+y, 
                         [self.recon_loss(x, y) \
                          for x, y in zip(self.encoder.hiddens, self.decoder.hiddens)])
        l = l_rec
        if y is not None:
            y_pred = self.mlp(h)
            l_ce = F.softmax_cross_entropy(y_pred, y)
            l += l_ce
        self.cleargrads()
        l.backward()
        self.optimizer_enc.update()
        self.optimizer_mlp.update()
        self.optimizer_dec.update()

        # Discriminator
        h = Variable(h.data)  # disconnect
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        hz = self.generator0(z)
        x_gen = self.decoder(hz)
        d_x_gen = self.discriminator(x_gen)
        d_x_real = self.discriminator(x)
        l_dis = self.lsgan_loss(d_x_gen, d_x_real)
        self.cleargrads()
        l_dis.backward()
        self.optimizer_dis.update()
        
        # Generator
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        hz = self.generator0(z)
        x_gen = self.decoder(hz)
        d_x_gen = self.discriminator(x_gen)
        h_gen = self.encoder(x_gen)
        l_gen = self.lsgan_loss(d_x_gen) + self.recon_loss(h, h_gen)
        self.cleargrads()
        l_gen.backward()
        l_dec.backward()
        self.optimizer_gen.update()
        self.optimizer_dec.update()
        

    def test(self, x_l, y_l, epoch, filename):
        """generate samples, then save"""
        x_gen = self.generate(x_l, test=True)
        self.save(x_l, x_gen, epoch, filename)

        d_x_gen = self.discriminator(x_gen, test=True)
        loss = self.lsgan_loss(d_x_gen)
        return loss
        
    def generate(self, x_l, test):
        h = self.encoder(x_l, test)
        xp = cuda.get_array_module(x_l)
        z = Variable(cuda.to_gpu(xp.random.rand(x_l.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.generator0(h, z, test)
        return x_gen

    def save(self, x_l, x_gen, epoch, filename):
        # Create dir path for genenrated images
        dpath_gen = os.path.join(filename,
                                 "gen", 
                                 "images_{:05d}".format(epoch))
        if os.path.exists(dpath_gen):
            shutil.rmtree(dpath_gen)
            os.makedirs(dpath_gen)
        else:
            os.makedirs(dpath_gen)

        # Create dir path for real images
        dpath_real = os.path.join(filename,
                                 "real", 
                                 "images_{:05d}".format(epoch))
        if os.path.exists(dpath_real):
            shutil.rmtree(dpath_real)
            os.makedirs(dpath_real)
        else:
            os.makedirs(dpath_real)

        # Save labeled images
        imgs = cuda.to_cpu(x_l.data)
        imgs = 127.5 * imgs + 127.5
        for i, img in enumerate(imgs):
            fpath = os.path.join(dpath_real, "{:05d}.png".format(i))
            cv2.imwrite(fpath, np.squeeze(img))

        # Save generated images
        imgs = cuda.to_cpu(x_gen.data)
        imgs = 127.5 * imgs + 127.5
        for i, img in enumerate(imgs):
            fpath = os.path.join(dpath_gen, "{:05d}.png".format(i))
            cv2.imwrite(fpath, np.squeeze(img))

    def serialize(self, epoch, filename):
        # Create dir path
        dpath = os.path.join(filename, "./model_{:05d}".format(epoch))
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
            os.makedirs(dpath)
        else:
            os.makedirs(dpath)

        # Serialize
        fpath = os.path.join(dpath, "encoder.h5py")
        serializers.save_hdf5(fpath, self.encoder)
        fpath = os.path.join(dpath, "generator.h5py")
        serializers.save_hdf5(fpath, self.generator)
        
    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.mlp.cleargrads()
        self.decoder.cleargrads()
        self.generator.cleargrads()
        self.discriminator.cleargrads()
        
