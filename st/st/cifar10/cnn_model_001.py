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
from st.utils import to_device
from st.mean_only_bn import BatchNormalization

class Model(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Model, self).__init__(
            # Conv/NIN/Linear
            conv0=L.Convolution2D(3, 128, ksize=3, stride=1, pad=1),
            conv1=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            conv2=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            conv3=L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            conv4=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv5=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv6=L.Convolution2D(256, 512, ksize=3, stride=1, pad=0),
            conv7=L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            conv8=L.Convolution2D(256, 128, ksize=1, stride=1, pad=0),
            #nin0=L.MLPConvolution2D(512, 256, ksize=3, stride=1, pad=0),
            #nin1=L.MLPConvolution2D(256, 128, ksize=3, stride=1, pad=0),
            linear=L.Linear(128, 10),
            
            # Batchnorm
            bn_conv0=BatchNormalization(128),
            bn_conv1=BatchNormalization(128),
            bn_conv2=BatchNormalization(128),
            bn_conv3=BatchNormalization(256),
            bn_conv4=BatchNormalization(256),
            bn_conv5=BatchNormalization(256),
            bn_conv6=BatchNormalization(512),
            bn_conv7=BatchNormalization(256),
            bn_conv8=BatchNormalization(128),
            #bn_linear=BatchNormalization(10),

            #bn_nin0=BatchNormalization(256),
            #bn_nin1=BatchNormalization(128),
        )
        self.act = F.leaky_relu
        
    def __call__(self, x, test=False):
        #TODO: gaussian noise
        
        # (conv -> act -> bn) x 3 -> maxpool -> dropout
        h = self.bn_conv0(self.act(self.conv0(x), 0.1), test)
        h = self.bn_conv1(self.act(self.conv1(h), 0.1), test)
        h = self.bn_conv2(self.act(self.conv2(h), 0.1), test)
        h = self.max_pooling_2d(h, (2, 2))  # 32 -> 16
        h = self.dropout(h, 0.5, not test)
        
        # (conv -> act -> bn) x 3 -> maxpool -> dropout
        h = self.bn_conv3(self.act(self.conv3(h), 0.1), test)
        h = self.bn_conv4(self.act(self.conv4(h), 0.1), test)
        h = self.bn_conv5(self.act(self.conv5(h), 0.1), test)
        h = self.max_pooling_2d(h, (2, 2))  # 16 -> 8
        h = self.dropout(h, 0.5, not test)
        
        # conv -> act -> bn -> (nin -> act -> bn) x 2
        h = self.bn_conv6(self.act(self.conv6(h), 0.1), test)  # 8 -> 7
        h = self.bn_conv7(self.act(self.conv7(h), 0.1), test)
        h = self.bn_conv8(self.act(self.conv8(h), 0.1), test)

        h = F.average_pooling_2d(h, (7, 7))
        h = self.linear(h)
        
        return h

