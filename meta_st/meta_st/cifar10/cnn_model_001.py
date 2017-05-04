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
from meta_st.utils import to_device
from meta_st.linear import Linear
from meta_st.convolution import Convolution2D
from meta_st.deconvolution import Deconvolution2D
from meta_st.batch_normalization import BatchNormalization

class ConvUnit(Chain):
    def __init__(self, maps, maps, k=4, s=2, p=1, act=F.relu):
        super(ConvUnit, self).__init__(
            conv=Convolution2D(maps, maps, ksize=k, stride=s, pad=p, nobias=True),
            bn=BatchNormalization(maps, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, h, model_params, test=False):
        h = self.conv(h, model_params["/conv/W"], )
        h = self.bn(h, 
                    model_params["/bn/gamma"], 
                    model_params["/bn/beta"],
                    test=test)
        h = self.act(h)
        return h

class ResConvUnit(Chain):
    def __init__(self, maps, act=F.relu):
        super(ResConvUnit, self).__init__(
            conv0=Convolution2D(maps, maps/2, ksize=1, stride=1, pad=0,
                                nobias=True),
            bn0=BatchNormalization(maps/2, decay=0.9, use_cudnn=True),
            conv1=Convolution2D(maps/2, maps/2, ksize=3, stride=1, pad=1, 
                                nobias=True),
            bn1=BatchNormalization(maps/2, decay=0.9, use_cudnn=True),
            conv2=Convolution2D(maps/2, maps, ksize=1, stride=1, pad=0, 
                                nobias=True),
            bn2=BatchNormalization(maps, decay=0.9, use_cudnn=True),
        )
        self.act = act
        
    def __call__(self, x, model_params, test=False):
        h = self.conv0(x, model_params["/conv0/W"])
        h = self.bn0(h, 
                     model_params["/bn0/gamma"], 
                     model_params["/bn0/beta"], 
                     test=test)
        h = self.act(h)

        h = self.conv1(h, model_params["/conv1/W"])
        h = self.bn1(h, 
                     model_params["/bn1/gamma"], 
                     model_params["/bn1/beta"], 
                     test=test)
        h = self.act(h)

        h = self.conv2(h, model_params["/conv2/W"])
        h = h + x
        h = self.bn2(h, 
                     model_params["/bn2/gamma"], 
                     model_params["/bn2/beta"], 
                     test=test)
        h = self.act(h)

        return h

class Model(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Model, self).__init__(
            convunit00=ConvUnit(3, 128, k=3, s=1, p=1, act=act),
            convunit01=ConvUnit(128, 128, k=3, s=1, p=1, act=act),
            convunit02=ConvUnit(128, 128, k=3, s=1, p=1, act=act),
            convunit10=ConvUnit(128, 256, k=3, s=1, p=1, act=act),
            convunit12=ConvUnit(256, 256, k=3, s=1, p=1, act=act),
            convunit12=ConvUnit(256, 256, k=3, s=1, p=1, act=act),
            convunit20=ConvUnit(256, 512, k=3, s=1, p=0, act=act),
            convunit22=ConvUnit(512, 256, k=1, s=1, p=0, act=act),
            convunit22=ConvUnit(256, 128, k=1, s=1, p=0, act=act),
            linear=Linear(128, 10),
        )
        self.act = act

    def __call__(self, x, model_params, test=False):
        # First block
        mp_filtered = self._filter_model_params(model_params, "/convunit00")
        h = self.convunit00(x, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/convunit01")
        h = self.convunit01(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/convunit02")
        h = self.convunit02(h, mp_filtered, test)
        h = F.max_pooling_2d(h, (2, 2))  # 32 -> 16
        h = F.dropout(h, train=not test)

        # Second block
        mp_filtered = self._filter_model_params(model_params, "/convunit10")
        h = self.convunit10(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/convunit11")
        h = self.convunit11(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/convunit12")
        h = self.convunit12(h, mp_filtered, test)
        h = F.max_pooling_2d(h, (2, 2))  # 16 -> 8
        h = F.dropout(h, train=not test)

        # Third block
        mp_filtered = self._filter_model_params(model_params, "/convunit20")
        h = self.convunit20(h, mp_filtered, test)  # 8 -> 6
        mp_filtered = self._filter_model_params(model_params, "/convunit21")
        h = self.convunit21(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/convunit22")
        h = self.convunit22(h, mp_filtered, test)
        h = F.average_pooling_2d(h, (6, 6))
        
        # Linear
        mp_filtered = self._filter_model_params(model_params, "/linear")
        y = self.linear(h, mp_filtered["/W"], mp_filtered["/b"])

        return y

    def _filter_model_params(self, model_params, name=""):
        """Filter parameter

        Result is parameters whose prefix of the ogirinal name is deleted, 
        the following name remains.
        """
        model_params_filtered = OrderedDict()
        for k, v in model_params.items():
            if not k.startswith(name):
                continue
            k = k.split("/")[2:]
            k = "/" + "/".join(k)
            model_params_filtered[k] = v
            
        return model_params_filtered
