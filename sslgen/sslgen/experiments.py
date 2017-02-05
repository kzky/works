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
from losses import ReconstructionLoss, NegativeEntropyLoss, GANLoss
from sklearn.metrics import confusion_matrix
from sslgen.cnn_model import Generator, Discriminator
        
class Experiment(object):

    def __init__(self, device=None, 
                 n_cls=10, dims=100, learning_rate=1e-3, act=F.relu):
        # Settings
        self.device = device
        self.n_cls = n_cls
        self.dims = dims
        self.act = act
        self.learning_rate = 1e-3

        # Model
        self.generator = Generator(device=device, act=act, n_cls=n_cls, dims=dims)
        self.generator.to_gpu(device) if self.device else None
        self.discriminator = Discriminator(device=device, act=act)
        self.discriminator.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer_gen = optimizers.Adam(learning_rate)
        self.optimizer_gen.setup(self.generator)
        self.optimizer_gen.use_cleargrads()
        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_dis.use_cleargrads()
        
        # Losses
        self.recon_loss = ReconstructionLoss()
        self.gan_loss = GANLoss()
        
    def train(self, x_l, y_l, x_u):
        # Train for labeled sampels
        self._train(x_l, y_l)

        # Train for unlabeled sampels
        self._train(x_u, None)

    def _train(self, x_real, y=None):
        bs = x_real.shape[0]
        
        # Train Discriminator
        z = self.generate_random(bs, self.dims)
        x_gen = self.generator(x_real, y, z)
        d_x_gen = self.discriminator(x_gen)
        d_x = self.discriminator(x_real)
        loss_dis = self.gan_loss(d_x_gen, d_x)
        self.generator.cleargrads()
        self.discriminator.cleargrads()
        loss_dis.backward()
        self.optimizer_dis.update()
        
        # Train Generator
        z = self.generate_random(bs, self.dims)
        x_gen = self.generator(x_real, y, z)
        d_x_gen = self.discriminator(x_gen)
        loss_gen = self.gan_loss(d_x_gen) + self.recon_loss(x_gen, x_real)
        self.generator.cleargrads()
        self.discriminator.cleargrads()
        loss_gen.backward()
        self.optimizer_gen.update()
        
    def test(self, x, y):
        # Generate Images
        bs = x.shape[0]
        z = self.generate_random(bs, self.dims)
        x_gen = self.generator(x, y, z)
        d_x_gen = self.discriminator(x_gen)

        # Save generated images
        if os.path.exists("./test_gen"):
            shutil.rmtree("./test_gen")
            os.mkdir("./test_gen")
        else:
            os.mkdir("./test_gen")

        x_gen_data = cuda.to_cpu(x_gen.data)
        for i, img in enumerate(x_gen_data):
            fpath = "./test_gen/{:05d}.png".format(i)
            cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)

        # D(x_gen) values
        d_x_gen_data = [float(data[0]) for data in cuda.to_cpu(d_x_gen.data)][0:100]

        return d_x_gen_data
        
    def save_model(self, epoch):
        dpath  = "./model"
        if not os.path.exists(dpath):
            os.makedirs(dpath)
            
        fpath = "./model/generator_{:05d}.h5py".format(epoch)
        serializers.save_hdf5(fpath, self.generator)

    def generate_random(self, bs, dims=30):
        r = np.random.uniform(-1, 1, (bs, dims)).astype(np.float32)
        r = to_device(r, self.device)
        return r
