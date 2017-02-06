"""Models
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
import logging
import time
from utils import to_device
from chainer_fix import BatchNormalization

class ResEnc(Chain):

    def __init__(self, inmap=32, outmap=64, act=F.relu, dn=False):
        if dn:
            super(ResEnc, self).__init__(
                conv0=L.Convolution2D(inmap, outmap, 4, stride=2, pad=1),
                conv1=L.Convolution2D(outmap, outmap, 3, stride=1, pad=1),
                conv2=L.Convolution2D(inmap, outmap, 4, stride=2, pad=1),
                bn0=L.BatchNormalization(outmap, decay=0.9),
                bn1=L.BatchNormalization(outmap, decay=0.9),
                bn2=L.BatchNormalization(outmap, decay=0.9)
            )
        else:
            super(ResEnc, self).__init__(
                conv0=L.Convolution2D(inmap, outmap, 3, stride=1, pad=1),
                conv1=L.Convolution2D(inmap, outmap, 3, stride=1, pad=1),
                bn0=L.BatchNormalization(inmap, decay=0.9),
                bn1=L.BatchNormalization(inmap, decay=0.9)
            )
            
        self.act = act
        self.dn = dn

    def __call__(self, x):
        h = self.conv0(x)
        h = self.bn0(h)
        h = self.act(h)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.act(h)

        h_s = x
        if self.dn:
            h_s = self.conv2(x)
            h_s = self.bn2(h_s)
            h_s = self.act(h_s)

        return h + h_s

class ResDec(Chain):

    def __init__(self, inmap=32, outmap=64, act=F.relu, up=False):
        if up:
            super(ResDec, self).__init__(
                decovn0=L.Deconvolution2D(inmap, outmap, 4, stride=2, pad=1),
                decovn1=L.Deconvolution2D(outmap, outmap, 3, stride=1, pad=1),
                decovn2=L.Deconvolution2D(inmap, outmap, 4, stride=2, pad=1),
                bn0=L.BatchNormalization(outmap, decay=0.9),
                bn1=L.BatchNormalization(outmap, decay=0.9),
                bn2=L.BatchNormalization(outmap, decay=0.9)
            )
        else:
            super(ResDec, self).__init__(
                decovn0=L.Deconvolution2D(inmap, outmap, 3, stride=1, pad=1),
                decovn1=L.Deconvolution2D(inmap, outmap, 3, stride=1, pad=1),
                bn0=L.BatchNormalization(inmap, decay=0.9),
                bn1=L.BatchNormalization(inmap, decay=0.9)
            )
            
        self.act = act
        self.up = up
        self.outmap = outmap

    def __call__(self, x):
        h = self.decovn0(x)
        h = self.bn0(h)
        h = self.act(h)
        h = self.decovn1(h)
        h = self.bn1(h)
        if self.outmap != 1:
            h = self.act(h)

        h_s = x
        if self.up:
            h_s = self.decovn2(x)
            h_s = self.bn2(h_s)
            if self.outmap != 1:
                h_s = self.act(h_s)

        return h + h_s
    
class Encoder(Chain):

    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            # Encoder
            resenc0=ResEnc(1, 32, act, dn=True),
            resenc1=ResEnc(32, 32, act),
            resenc2=ResEnc(32, 64, act, dn=True),
            resenc3=ResEnc(64, 64, act),
            linear0=L.Linear(64 * 7 * 7, 32),
            linear1=L.Linear(32, 10),
            bn0=L.BatchNormalization(32, decay=0.9),
            # BranchNet
            linear0_bn=L.Linear(32*14*14, 10),
            linear1_bn=L.Linear(64*7*7, 10),
            linear2_bn=L.Linear(32, 10),
        )

        self.act = act
        self.hiddens = []
        self.classifiers = []
        
    def __call__(self, x, test=False):
        self.hiddens = []
        self.classifiers = []

        h = self.resenc0(x)  # 14x14
        self.hiddens.append(h)
        h = self.resenc1(h) # 14x14
        self.hiddens.append(h)
        h = self.resenc2(h)  # 7x7
        self.hiddens.append(h)
        h = self.resenc3(h) # 7x7
        self.hiddens.append(h)
        
        h = self.linear0(h)
        h = self.bn0(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        return h

class Decoder(Chain):

    def __init__(self, act=F.relu):
        super(Decoder, self).__init__(
            # Decoer
            linear0=L.Linear(10, 32),
            linear1=L.Linear(32, 64 * 7 * 7),
            bn0=L.BatchNormalization(32, decay=0.9),
            bn1=L.BatchNormalization(64 * 7 * 7, decay=0.9),
            resdec0=ResDec(64, 64, act),
            resdec1=ResDec(64, 32, act, up=True),
            resdec2=ResDec(32, 32, act),
            resdec3=ResDec(32, 1, act, up=True),
            # BranchNet
            linear0_bn=L.Linear(32, 10),
            linear1_bn=L.Linear(64*7*7, 10),
            linear2_bn=L.Linear(32*14*14, 10),
        )
                
        self.act = act
        self.hiddens = []
        self.classifiers = []

    def __call__(self, y, test=False):
        self.hiddens = []
        self.classifiers = []
        bs = y.shape[0]

        h = self.linear0(y)
        h = self.bn0(h)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.act(h)

        h = F.reshape(h, (bs, 64, 7, 7))
        self.hiddens.append(h)
        h = self.resdec0(h)  #7x7
        self.hiddens.append(h)
        h = self.resdec1(h)  #14x14
        self.hiddens.append(h)
        h = self.resdec2(h)  #14x14
        self.hiddens.append(h)
        h = self.resdec3(h)  #28x28
        return h

class AutoEncoder(Chain):
    
    def __init__(self, act=F.relu):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act)
        )
        


    
