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
from mlp_model import AutoEncoder
from losses import ReconstructionLoss, NegativeEntropyLoss
from sklearn.metrics import confusion_matrix

class Experiment(object):

    def __init__(self, device=None, learning_rate=1e-3, act=F.relu):

        # Settings
        self.device = device
        self.act = act
        self.learning_rate = 1e-3

        # Model
        self.ae = AutoEncoder(act)
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
        # TODO: add hiddens
        y_prob = F.softmax(y)
        x_recon = self.ae.decoder(y_prob)
        loss_recon_l = self.recon_loss(x_recon, x_l)
        
        # unlabeled loss
        y = self.ae.encoder(x_u)
        loss_ne_u = self.ne_loss(y)
        # TODO: add hiddens
        y_prob = F.softmax(y)
        x_recon = self.ae.decoder(y_prob)
        loss_recon_u = self.recon_loss(x_recon, x_u)

        # sum losses
        loss = loss_ce + loss_ne_l + loss_recon_l + loss_ne_u + loss_recon_u

        # Backward and Update
        self.ae.cleargrads()
        loss.backward()
        self.optimizer.update()
        
    def test(self, epoch, x_l, y_l):
        y = self.ae.encoder(x_l, test=True)
        self.generate_and_save_wrong_samples(x_l, y_l, y)
        self.save_model(epoch)
        acc = F.accuracy(y, y_l)
        return acc

    def generate_and_save_wrong_samples(self, x_l, y_l, y):
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
        self.save_incorrect_info(x_rec_cpu, x_l.data[idx, ],
                                 y.data[idx, ], y_l.data[idx, ])

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
            cv2.imwrite(fpath, img.reshape(28, 28) * 127.5 + 127.5)
            
        # Images
        for i, img in enumerate(x_l):
            fpath = "./test/{:05d}.png".format(i)
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
        else:
            shutil.rmtree(dpath)
            os.makedirs(dpath)
            
        fpath = "./model/auto_encoder_{:05d}.h5py".format(epoch)
        serializers.save_hdf5(fpath, self.ae)

