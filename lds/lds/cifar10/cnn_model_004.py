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

class ConvUnit(Chain):
    def __init__(self, maps, act=F.relu):
        super(ConvUnit, self).__init__(
            conv0=L.Convolution2D(maps, maps, 3, stride=1, pad=1),
            bn0=L.BatchNormalization(maps, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, hiddens, test=False):
        h = self.conv0(h)
        h = self.bn0(h, test)
        h = self.act(h)

        return h

class ConvUnitPoolFinetune(Chain):
    def __init__(self, maps, act):
        super(ConvUnitPoolFinetune, self).__init__(
            conv_unit=ConvUnit(maps, act),
            conv=L.Convolution2D(maps, maps*2, 3, stride=1, pad=0),
            bn=L.BatchNormalization(maps*2, decay=0.9, use_cudnn=True),
        )
        self.act = act
    
    def __call__(self, h, hiddens, test=False):
        h = self.conv_unit(h, hiddens, test)
        hiddens.append(h)
        h = F.max_pooling_2d(h, (2, 2))
        h = self.conv(h)
        h = self.bn(h, test)
        h = self.act(h)
        return h

class Encoder(Chain):
    """Spatial size is reduced by maxpooling.
    """
    def __init__(self, act=F.relu):
        super(Encoder, self).__init__(
            conv0=L.Convolution2D(3, 32, 3, stride=1, pad=1),
            bn0=L.BatchNormalization(32, decay=0.9, use_cudnn=True),
            block0=ConvUnitPoolFinetune(32, act),
            block1=ConvUnitPoolFinetune(64, act),
            block2=ConvUnitPoolFinetune(128, act),
        )

        self.act = act
        self.hiddens = []
        
    def __call__(self, x, test=False):
        self.hiddens = []

        # Input
        h = self.conv0(x)
        h = self.bn0(h, test)
        h = self.act(h)
        self.hiddens.append(h)

        # Blocks
        h = self.block0(h, self.hiddens, test)
        self.hiddens.append(h)
        h = self.block1(h, self.hiddens, test)
        self.hiddens.append(h)
        h = self.block2(h, self.hiddens, test)
        
        return h

class MLP(Chain):
    def __init__(self, act=F.relu):
        super(MLP, self).__init__(
            linear0=L.Linear(256*4*4, 10),
            #bn0=L.BatchNormalization(256, decay=0.9, use_cudnn=True),
        )
        self.act = act

    def __call__(self, h, test=False):
        #h = F.average_pooling_2d(h, (4, 4))
        #h = self.bn0(h, test)
        h = self.linear0(h)
        return h
    
class DeconvUnit(Chain):

    def __init__(self, maps, act=F.relu):
        super(DeconvUnit, self).__init__(
            deconv0=L.Convolution2D(maps, maps/2, 3, stride=1, pad=1),
            bn0=L.BatchNormalization(maps/2, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, hiddens, test=False):
        h = self.deconv0(h)
        h = self.bn0(h, test)
        h = self.act(h)
        return h

class DeconvUnitPoolFinetune(Chain):
    def __init__(self, maps, act):
        super(DeconvUnitPoolFinetune, self).__init__(
            deconv_unit=DeconvUnit(maps, act),
            deconv_pool=L.Deconvolution2D(maps/2, maps/2, 4, stride=2, pad=1),
            bn_pool=L.BatchNormalization(maps/2, decay=0.9, use_cudnn=True),
            deconv=L.Deconvolution2D(maps/2, maps/2, 1, stride=1, pad=0),
            bn=L.BatchNormalization(maps/2, decay=0.9, use_cudnn=True),
        )
        self.act = act

    def __call__(self, h, hiddens, test=False):
        h = self.deconv_unit(h, hiddens, test)
        h = self.deconv_pool(h)
        h = self.bn_pool(h, test)
        h = self.act(h)  # pooling is not actual pooling, so take activation here.
        hiddens.append(h)
        h = self.deconv(h)
        h = self.bn(h, test)
        h = self.act(h)
        return h

class Decoder(Chain):
    """Unpooling
    """
    def __init__(self, act=F.relu):
        super(Decoder, self).__init__(
            block0=DeconvUnitPoolFinetune(256, act),
            block1=DeconvUnitPoolFinetune(128, act),
            block2=DeconvUnitPoolFinetune(64, act),
            deconv=L.Deconvolution2D(32, 3, 3, stride=1, pad=1),
        )
                
        self.act = act
        self.hiddens = []

    def __call__(self, h, test=False):
        self.hiddens = []

        h = self.block0(h, self.hiddens, test)
        self.hiddens.append(h)

        h = self.block1(h, self.hiddens, test)
        self.hiddens.append(h)

        h = self.block2(h, self.hiddens, test)
        self.hiddens.append(h)

        h = self.deconv(h)
        return h

class AutoEncoderWithMLP(Chain):
    
    def __init__(self, act=F.relu):
        super(AutoEncoderWithMLP, self).__init__(
            encoder=Encoder(act=act),
            decoder=Decoder(act=act),
            mlp=MLP(act),
        )
        


