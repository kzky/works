"""FCNN model
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
    def __init__(self, imaps, omaps, k=4, s=2, p=1, act=F.relu):
        super(ConvUnit, self).__init__(
            conv=Convolution2D(imaps, omaps, ksize=k, stride=s, pad=p, nobias=True),
            bn=BatchNormalization(omaps, decay=0.9, use_cudnn=True),
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

class Model(Chain):

    def __init__(self, device=None, act=F.leaky_relu):
        
        super(Model, self).__init__(
            # ConvBlock0
            conv00=ConvUnit(3, 96, k=3, s=1, p=1),
            conv01=ConvUnit(96, 96, k=3, s=1, p=1),
            conv02=ConvUnit(96, 96, k=3, s=1, p=1),
            bn0=BatchNormalization(96),
            # ConvBlock1
            conv10=ConvUnit(96, 192, k=3, s=1, p=1),
            conv11=ConvUnit(192, 192, k=3, s=1, p=1),
            conv12=ConvUnit(192, 192, k=3, s=1, p=1),
            bn1=BatchNormalization(192),
            # ConvBlock3
            conv20=ConvUnit(192, 192, k=3, s=1, p=0),
            conv21=ConvUnit(192, 192, k=1, s=1, p=0),
            conv22=ConvUnit(192, 10, k=1, s=1, p=0),
            bn2=BatchNormalization(10)
        )
        self.act = act

    def __call__(self, x, model_params, test=False):
        mp_filtered = self._filter_model_params(model_params, "/conv00")
        h = self.conv00(x, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/conv01")
        h = self.conv01(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/conv02")
        h = self.conv02(h, mp_filtered, test)
        h = F.max_pooling_2d(h, (2, 2))  # 32 -> 16
        mp_filtered = self._filter_model_params(model_params, "/bn0")
        h = self.bn0(h, 
                     mp_filtered["/gamma"], 
                     mp_filtered["/beta"],
                     test)
    
        mp_filtered = self._filter_model_params(model_params, "/conv10")
        h = self.conv10(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/conv11")
        h = self.conv11(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/conv12")
        h = self.conv12(h, mp_filtered, test)
        h = F.max_pooling_2d(h, (2, 2))  # 16 -> 8
        mp_filtered = self._filter_model_params(model_params, "/bn1")
        h = self.bn1(h, 
                     mp_filtered["/gamma"], 
                     mp_filtered["/beta"],
                     test)

        mp_filtered = self._filter_model_params(model_params, "/conv20")
        h = self.conv20(h, mp_filtered, test)  # 8 -> 6
        mp_filtered = self._filter_model_params(model_params, "/conv21")
        h = self.conv21(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "/conv22")
        h = self.conv22(h, mp_filtered, test)
        h = F.average_pooling_2d(h, (6, 6))  # 6 -> 1
        mp_filtered = self._filter_model_params(model_params, "/bn2")
        h = self.bn2(h, 
                     mp_filtered["/gamma"], 
                     mp_filtered["/beta"],
                     test)
        h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
        
        return h
    
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
