"""Small Models same as MNIST
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
from lds.utils import to_device
from lds.chainer_fix import BatchNormalization

class Encoder(Chain):

    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            # Encoder
            conv0=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv2=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            linear0=L.Linear(128 * 8 * 8, 64),
            linear1=L.Linear(64, 10),

            bn_conv0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_conv1=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_conv2=L.BatchNormalization(128, decay=0.9, use_cudnn=True),
            bn_conv3=L.BatchNormalization(128, decay=0.9, use_cudnn=True),
            bn_linear0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
        )

        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []
        
        # Convolution
        h = self.conv0(x)
        h = self.bn_conv0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv1(h)
        h = self.bn_conv1(h, test)
        h = self.act(h)
        self.hiddens.append(h)
        h = F.max_pooling_2d(h, (2, 2))  # 32x32 -> 16x16

        h = self.conv2(h)
        h = self.bn_conv2(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv3(h)
        h = self.bn_conv3(h, test)
        h = self.act(h)
        self.hiddens.append(h)
        h = F.max_pooling_2d(h, (2, 2))  # 16x16 -> 8x8

        # Linear
        h = self.linear0(h)   # 8x8 -> 64
        h = self.bn_linear0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        return h

class Decoder(Chain):
    """Unpooling
    """
    def __init__(self, act=F.relu):
        super(Decoder, self).__init__(
            linear0=L.Linear(10, 64),
            linear1=L.Linear(64, 128 * 8 * 8),
            deconv0=L.Deconvolution2D(128, 128, 2, stride=2, pad=0),
            deconv1=L.Deconvolution2D(128, 128, 3, stride=1, pad=1),
            deconv2=L.Deconvolution2D(128, 128, 3, stride=1, pad=1),
            deconv3=L.Deconvolution2D(128, 64, 2, stride=2, pad=0),
            deconv4=L.Deconvolution2D(64, 64, 3, stride=1, pad=1),
            deconv5=L.Deconvolution2D(64, 3, 3, stride=1, pad=1),
            
            bn_linear0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_linear1=L.BatchNormalization(128 * 8 * 8, decay=0.9, use_cudnn=True),
            bn_deconv0=L.BatchNormalization(128, decay=0.9, use_cudnn=True),
            bn_deconv1=L.BatchNormalization(128, decay=0.9, use_cudnn=True),
            bn_deconv2=L.BatchNormalization(128, decay=0.9, use_cudnn=True),
            bn_deconv3=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_deconv4=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
        )
                
        self.act = act
        self.hiddens = []

    def __call__(self, y, test=False):
        self.hiddens = []

        # Linear
        h = self.linear0(y)
        h = self.bn_linear0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.linear1(h)
        h = self.bn_linear1(h, test)
        h = self.act(h)
        bs = h.shape[0]
        d = h.shape[1]
        h = F.reshape(h, (bs, 128, 8, 8))
        

        # Deconvolution
        h = self.deconv0(h)
        h = self.bn_deconv0(h, test)  # 8x8 -> 16x16
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv1(h)
        h = self.bn_deconv1(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv2(h)
        h = self.bn_deconv2(h, test)
        h = self.act(h)

        h = self.deconv3(h)  # 16x16 -> 32x32
        h = self.bn_deconv3(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv4(h)
        h = self.bn_deconv4(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv5(h)
        return h

class AutoEncoder(Chain):
    
    def __init__(self, act=F.relu):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act)
        )
        


    
