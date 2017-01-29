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

def create_ae_experiment(model, act=F.relu):
    if model == "mlp":
        from mlp_model import AutoEncoder
        return AutoEncoder(act)

    if model == "cnn":
        from cnn_model import AutoEncoder
        return AutoEncoder(act)

def create_gan_experiment(model, act=F.relu, dim_rand=30):
    if model == "mlp":
        from mlp_model import Generator, Discriminator
        return Generator(act, dim_rand), Discriminator(act)
        
class AEExperiment(object):

    def __init__(self, device=None, model=None, learning_rate=1e-3, act=F.relu):

        # Settings
        self.device = device
        self.act = act
        self.learning_rate = 1e-3

        # Model
        self.ae = create_ae_experiment(model, act)
        self.ae.to_gpu(device) if self.device else None

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.ae)
        self.optimizer.use_cleargrads()
        
        # Losses
        self.recon_loss = ReconstructionLoss()
        self.ne_loss = NegativeEntropyLoss()
        
    def train(self, x_l, y_l, x_u):
        # Forward
        # labeled loss
        y = self.ae.encoder(x_l)
        loss_ce = F.softmax_cross_entropy(y, y_l)
        loss_ne_l = self.ne_loss(y)
        y_prob = F.softmax(y)
        x_recon = self.ae.decoder(y_prob)
        loss_recon_l = self.reconstruction(x_recon, x_l, 
                                           self.ae.encoder.hiddens, 
                                           self.ae.decoder.hiddens)
        
        # unlabeled loss
        y = self.ae.encoder(x_u)
        loss_ne_u = self.ne_loss(y)
        y_prob = F.softmax(y)
        x_recon = self.ae.decoder(y_prob)
        loss_recon_u = self.reconstruction(x_recon, x_u, 
                                           self.ae.encoder.hiddens, 
                                           self.ae.decoder.hiddens)

        # sum losses
        loss = loss_ce + loss_ne_l + loss_recon_l + loss_ne_u + loss_recon_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()

    def reconstruction(self, x_recon, x, hiddens_enc, hiddens_dec):
        loss_recon = self.recon_loss(x_recon, x)
        for h_enc, h_dec in zip(hiddens_enc, hiddens_dec[::-1]):
            loss_recon += self.recon_loss(h_enc, h_dec)
        return loss_recon
        
    def test(self, epoch, x_l, y_l):
        y = self.ae.encoder(x_l, test=True)
        self.generate_and_save_wrong_samples(x_l, y_l, y)
        self.save_model(epoch)
        acc = F.accuracy(y, y_l)
        return acc

    def generate_and_save_wrong_samples(self, x_l, y_l, y):
        y = F.softmax(y)
        y_argmax = F.argmax(y, axis=1)
        y_l_cpu = cuda.to_cpu(y_l.data)
        y_argmax_cpu = cuda.to_cpu(y_argmax.data)

        # Confuction Matrix
        cm = confusion_matrix(y_l_cpu, y_argmax_cpu)
        print(cm)

        # Wrong samples
        idx = np.where(y_l_cpu != y_argmax_cpu)[0]

        # Generate and Save
        x_rec = self.ae.decoder(y[idx, ], test=True)
        x_rec_cpu = cuda.to_cpu(x_rec.data)
        x_l_cpu = cuda.to_cpu(x_l.data)
        y_cpu = cuda.to_cpu(y.data)
        self.save_incorrect_info(x_rec_cpu, x_l_cpu[idx, ],
                                 y_cpu[idx, ], y_l_cpu[idx, ])

    def save_incorrect_info(self, x_rec, x_l, y, y_l):
        # Generated Images
        if os.path.exists("./test_gen"):
            shutil.rmtree("./test_gen")
            os.mkdir("./test_gen")
        else:
            os.mkdir("./test_gen")
     
        # Images
        if os.path.exists("./test"):
            shutil.rmtree("./test")
            os.mkdir("./test")
        else:
            os.mkdir("./test")
            
        # Generated Images
        for i, img in enumerate(x_rec):
            fpath = "./test_gen/{:05d}.png".format(i)
            if len(img.shape) == 1:
                cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)
            else:
                                
                cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)
            
        # Images
        for i, img in enumerate(x_l):
            fpath = "./test/{:05d}.png".format(i)
            if len(img.shape) == 1:
                cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)
            else:
                cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)
     
        # Label and Probability
        with open("./label_prediction.out", "w") as fpout:
            header = ["idx", "true", "pred"]
            header += ["prob_{}".format(i) for i in range(len(y[0]))]
            writer = csv.writer(fpout, delimiter=",")
            writer.writerow(header)
            for i, elm in enumerate(zip(y, y_l)):
                y_, y_l_ = elm
                row = [i] \
                      + [y_l_] \
                      + [np.argmax(y_)] \
                      + map(lambda x: "{:05f}".format(x) , y_.tolist())
                writer.writerow(row)

    def save_model(self, epoch):
        dpath  = "./model"
        if not os.path.exists(dpath):
            os.makedirs(dpath)
            
        fpath = "./model/decoder_{:05d}.h5py".format(epoch)
        serializers.save_hdf5(fpath, self.ae.decoder)

class GANExperiment(object):

    def __init__(self, decoder, device=None, model=None,
                 dim_rand=30, n_cls=10, learning_rate=1e-3, act=F.relu):

        # Settings
        self.device = device
        self.dim_rand = dim_rand
        self.n_cls = n_cls
        self.act = act
        self.learning_rate = 1e-5

        # Model
        generator, discriminator = create_gan_experiment(
            model, act=act, dim_rand=dim_rand)
        self.generator = generator
        self.generator.to_gpu(device) if self.device else None
        self.discriminator = discriminator
        self.discriminator.to_gpu(device) if self.device else None
        self.decoder = decoder

        # Optimizer
        self.optimizer_gen = optimizers.Adam(learning_rate)
        self.optimizer_gen.setup(self.generator)
        self.optimizer_gen.use_cleargrads()

        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_dis.use_cleargrads()

        # Losses
        self.gan_loss = GANLoss()
        
    def train(self, x_real, ):
        bs = x_real.shape[0]
                          
        # Train discriminator
        x_recon = self.generate_x_recon(bs)
        d_x = self.discriminator(x_real)
        z = self.generate_random(bs, self.dim_rand)
        x_gen = self.generator(x_recon, z)
        d_x_gen = self.discriminator(x_gen)
        loss = self.gan_loss(d_x_gen, d_x)
        self.discriminator.cleargrads()
        self.generator.cleargrads()
        loss.backward()
        self.optimizer_dis.update()

        # Train generator
        x_recon = self.generate_x_recon(bs)
        z = self.generate_random(bs, self.dim_rand)
        x_gen = self.generator(x_recon, z)
        d_x_gen = self.discriminator(x_gen)
        loss = self.gan_loss(d_x_gen)
        self.discriminator.cleargrads()
        self.generator.cleargrads()
        loss.backward()
        self.optimizer_gen.update()

    def test(self, epoch, bs):
        z = self.generate_random(bs, self.dim_rand)
        x_recon = self.generate_x_recon(bs)
        x_gen = self.generator(x_recon, z)

        # Generated Images
        dpath = "./gen/{:05d}".format(epoch)
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
            os.makedirs(dpath)
        else:
            os.makedirs(dpath)
            
        for i, img in enumerate(x_gen.data):
            fpath = "./gen/{:05d}/{:05d}.png".format(epoch, i)
            if len(img.shape) == 1:
                cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)
            else:
                cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)

        # Save model
        self.save_model(epoch)

    def generate_random_onehot(self, bs):
        y = np.zeros((bs, self.n_cls))
        cls = np.random.choice(self.n_cls, bs)
        y[np.arange(bs), cls] = 1.0
        y = y.astype(np.float32)
        return y

    def generate_random_prob(self, bs):
        pass
            
    def generate_x_recon(self, bs):
        #TODO: consider diversity, now only r \in {0, 1}
        y = self.generate_random_onehot(bs)
        y = to_device(y, self.device)
        x_recon = self.decoder(y)
        return x_recon

    def generate_random(self, bs, dim=30):
        r = np.random.uniform(-1, 1, (bs, dim)).astype(np.float32)
        r = to_device(r)
        return r

    def save_model(self, epoch):
        dpath  = "./model"
        if not os.path.exists(dpath):
            os.makedirs(dpath)
            
        fpath = "./model/generator_{:05d}.h5py".format(epoch)
        serializers.save_hdf5(fpath, self.generator)

    
