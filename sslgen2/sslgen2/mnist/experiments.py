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
        self.discriminator = Discriminator(device, act)

        self.encoder.to_gpu(device) if self.device else None
        self.decoder.to_gpu(device) if self.device else None
        self.generator.to_gpu(device) if self.device else None
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
        x_rec = self.decoder(h)
        l_rec = self.recon_loss(x, x_rec)
        self.cleargrads()
        l_rec.backward()
        self.optimizer_enc.update()
        self.optimizer_dec.update()

        # Discriminator
        h = Variable(h.data)  # disconnect
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.generator(h, z)
        d_x_gen = self.discriminator(x_gen)
        d_x_real = self.discriminator(x)
        l_dis = self.lsgan_loss(d_x_gen, d_x_real)
        self.cleargrads()
        l_dis.backward()
        self.optimizer_dis.update()
        
        # Generator
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.generator(h, z)
        d_x_gen = self.discriminator(x_gen)
        h_gen = self.encoder(x_gen)
        l_gen = self.lsgan_loss(d_x_gen) + self.recon_loss(h, h_gen)
        self.cleargrads()
        self.optimizer_gen.update()

    def test(self, x_l, y_l, epoch):
        """generate samples, then save"""
        x_gen = self.generate(x_l, test=True)
        self.save(x_gen, epoch)

        d_x_gen = self.discriminator(x_gen, test=True)
        loss = self.lsgan_loss(d_x_gen)
        return loss
        
    def generate(self, x_l, test):
        h = self.encoder(x_l, test)
        xp = cuda.get_array_module(x_l)
        z = Variable(cuda.to_gpu(xp.random.rand(x_l.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.generator(h, z, test)
        return x_gen

    def save(self, x_gen, epoch):
        # Create dir path
        dpath = "./images_{:05d}".format(epoch)
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
            os.makedirs(dpath)
        else:
            os.makedirs(dpath)

        # Save
        imgs = cuda.to_cpu(x_gen.data)
        imgs = 127.5 * imgs + 127.5
        for i, img in enumerate(imgs):
            fpath = os.path.join(dpath, "_{:05d}.png".format(i))
            cv2.imwrite(fpath, np.squeeze(img))

    def serialize(self, epoch):
        # Create dir path
        dpath = "./model_{:05d}".format(epoch)
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
            os.makedirs(dpath)
        else:
            os.makedirs(dpath)

        # Serialize
        fpath = os.path.join(dpath, "encoder.h5py")
        serializers.save_hdf5(fpath, self.encoder)
        fpath = os.path.join(dpath, "decoder.h5py")
        serializers.save_hdf5(fpath, self.generator)
        
    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.generator.cleargrads()
        self.discriminator.cleargrads()

class Experiment001(Experiment000):
    """Enc-Dec, Enc-Gen-Enc, Enc-Gen-Dis.

    Decoder and Genrator shares parameters.
    Update Generator0 only when training generator, i.e., not train Decoder.
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
        from sslgen2.mnist.cnn_model_001 \
            import Encoder, Decoder, Generator0, Discriminator
        self.encoder = Encoder(device, act)
        self.decoder = Decoder(device, act)
        self.generator0 = Generator0(dim, device, act)
        self.discriminator = Discriminator(device, act)

        self.encoder.to_gpu(device) if self.device else None
        self.decoder.to_gpu(device) if self.device else None
        self.generator0.to_gpu(device) if self.device else None
        self.discriminator.to_gpu(device) if self.device else None
        
        # Optimizer
        self.optimizer_enc = optimizers.Adam(learning_rate)
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_dec = optimizers.Adam(learning_rate)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dec.use_cleargrads()
        self.optimizer_gen = optimizers.Adam(learning_rate)
        self.optimizer_gen.setup(self.generator0)
        self.optimizer_gen.use_cleargrads()
        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_dis.use_cleargrads()

    def train(self, x):
        # Encoder/Decoder
        h = self.encoder(x)
        x_rec = self.decoder(h)
        l_rec = self.recon_loss(x, x_rec)
        self.cleargrads()
        l_rec.backward()
        self.optimizer_enc.update()
        self.optimizer_dec.update()

        # Discriminator
        h = Variable(h.data)  # disconnect
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.decoder(self.generator0(z))
        d_x_gen = self.discriminator(x_gen)
        d_x_real = self.discriminator(x)
        l_dis = self.lsgan_loss(d_x_gen, d_x_real)
        self.cleargrads()
        l_dis.backward()
        self.optimizer_dis.update()
        
        # Generator
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.decoder(self.generator0(z))
        d_x_gen = self.discriminator(x_gen)
        h_gen = self.encoder(x_gen)
        l_gen = self.lsgan_loss(d_x_gen) + self.recon_loss(h, h_gen)
        self.cleargrads()
        self.optimizer_gen.update()
        
    def generate(self, x_l, test):
        xp = cuda.get_array_module(x_l)
        z = Variable(cuda.to_gpu(xp.random.rand(x_l.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.decoder(self.generator0(z, test))
        return x_gen

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.generator0.cleargrads()
        self.discriminator.cleargrads()

    def serialize(self, epoch):
        # Create dir path
        dpath = "./model_{:05d}".format(epoch)
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
            os.makedirs(dpath)
        else:
            os.makedirs(dpath)

        # Serialize
        fpath = os.path.join(dpath, "encoder.h5py")
        serializers.save_hdf5(fpath, self.encoder)
        fpath = os.path.join(dpath, "decoder.h5py")
        serializers.save_hdf5(fpath, self.generator0)

class Experiment002(Experiment001):
    """Enc-Dec, Enc-Gen-Enc, Enc-Gen-Dis.

    Decoder and Genrator shares parameters.
    Update Generator0 and Decoder when training generator.
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
        from sslgen2.mnist.cnn_model_001 \
            import Encoder, Decoder, Generator0, Discriminator
        self.encoder = Encoder(device, act)
        self.decoder = Decoder(device, act)
        self.generator0 = Generator0(dim, device, act)
        self.discriminator = Discriminator(device, act)

        self.encoder.to_gpu(device) if self.device else None
        self.decoder.to_gpu(device) if self.device else None
        self.generator0.to_gpu(device) if self.device else None
        self.discriminator.to_gpu(device) if self.device else None
        
        # Optimizer
        self.optimizer_enc = optimizers.Adam(learning_rate)
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_dec = optimizers.Adam(learning_rate)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dec.use_cleargrads()
        self.optimizer_gen0 = optimizers.Adam(learning_rate)
        self.optimizer_gen0.setup(self.generator0)
        self.optimizer_gen0.use_cleargrads()
        self.optimizer_gen1 = optimizers.Adam(learning_rate)
        self.optimizer_gen1.setup(self.decoder)
        self.optimizer_gen1.use_cleargrads()
        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_dis.use_cleargrads()

        
    def train(self, x):
        # Encoder/Decoder
        h = self.encoder(x)
        x_rec = self.decoder(h)
        l_rec = self.recon_loss(x, x_rec)
        self.cleargrads()
        l_rec.backward()
        self.optimizer_enc.update()
        self.optimizer_dec.update()

        # Discriminator
        h = Variable(h.data)  # disconnect
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.decoder(self.generator0(z))
        d_x_gen = self.discriminator(x_gen)
        d_x_real = self.discriminator(x)
        l_dis = self.lsgan_loss(d_x_gen, d_x_real)
        self.cleargrads()
        l_dis.backward()
        self.optimizer_dis.update()
        
        # Generator
        xp = cuda.get_array_module(x)
        z = Variable(cuda.to_gpu(xp.random.rand(x.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.decoder(self.generator0(z))
        d_x_gen = self.discriminator(x_gen)
        h_gen = self.encoder(x_gen)
        l_gen = self.lsgan_loss(d_x_gen) + self.recon_loss(h, h_gen)
        self.cleargrads()
        self.optimizer_gen0.update()
        self.optimizer_gen1.update()
        
    def generate(self, x_l, test):
        xp = cuda.get_array_module(x_l)
        z = Variable(cuda.to_gpu(xp.random.rand(x_l.shape[0], self.dim).astype(xp.float32), self.device))
        x_gen = self.decoder(self.generator0(z, test))
        return x_gen
