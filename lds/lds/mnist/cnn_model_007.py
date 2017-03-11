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
from lds.utils import to_device
from lds.chainer_fix import BatchNormalization

class Encoder(Chain):

    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            # Encoder
            conv0=L.Convolution2D(1, 32, 3, stride=1, pad=1),
            conv1=L.Convolution2D(32, 32, 3, stride=1, pad=1),
            conv2=L.Convolution2D(32, 32, 4, stride=2, pad=1),
            conv3=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            conv4=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv5=L.Convolution2D(64, 64, 4, stride=2, pad=1),
            linear0=L.Linear(64 * 7 * 7, 32),
            linear1=L.Linear(32, 10),

            bn_conv0=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
            bn_conv1=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
            bn_conv2=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
            bn_conv3=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_conv4=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_conv5=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_linear0=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
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

        h = self.conv2(h)  # 28x28 -> 14x14
        h = self.bn_conv2(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv3(h)
        h = self.bn_conv3(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.conv4(h)
        h = self.bn_conv4(h, test)
        h = self.act(h)
        self.hiddens.append(h)
        
        h = self.conv5(h)  # 14x14 -> 7x7
        h = self.bn_conv5(h)
        h = self.act(h)
        self.hiddens.append(h)

        # Linear
        h = self.linear0(h)   # 128x7x7 -> 64
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
            linear0=L.Linear(10, 32),
            linear1=L.Linear(32, 64 * 7 * 7),
            deconv0=L.Deconvolution2D(64, 64, 2, stride=2, pad=0),
            deconv1=L.Deconvolution2D(64, 64, 3, stride=1, pad=1),
            deconv2=L.Deconvolution2D(64, 64, 3, stride=1, pad=1),
            deconv3=L.Deconvolution2D(64, 32, 2, stride=2, pad=0),
            deconv4=L.Deconvolution2D(32, 32, 3, stride=1, pad=1),
            deconv5=L.Deconvolution2D(32, 1, 3, stride=1, pad=1),
            
            bn_linear0=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
            bn_linear1=L.BatchNormalization(64 * 7 * 7, decay=0.9, use_cudnn=True),
            bn_deconv0=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_deconv1=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_deconv2=L.BatchNormalization(64, decay=0.9, use_cudnn=True),
            bn_deconv3=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
            bn_deconv4=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
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
        h = F.reshape(h, (bs, 64, 7, 7))
        self.hiddens.append(h)

        # Deconvolution
        h = self.deconv0(h)
        h = self.bn_deconv0(h, test)  # 7x7 -> 14x14
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv1(h)
        h = self.bn_deconv1(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv2(h)
        h = self.bn_deconv2(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        h = self.deconv3(h)  # 14x14 -> 28x28
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
        


    
