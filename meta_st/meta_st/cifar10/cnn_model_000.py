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
                    model_params["/conv/gamma"], 
                    model_params["/conv/beta"],
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
        h = self.conv0(x, model_params["conv0/W"])
        h = self.bn0(h, 
                     model_params["bn0/gamma"], 
                     model_params["bn0/beta"], 
                     test=test)
        h = self.act(h)

        h = self.conv1(h, model_params["conv1/W"])
        h = self.bn1(h, 
                     model_params["bn1/gamma"], 
                     model_params["bn1/beta"], 
                     test=test)
        h = self.act(h)

        h = self.conv2(h, model_params["conv2/W"])
        h = h + x
        h = self.bn2(h, 
                     model_params["bn2/gamma"], 
                     model_params["bn2/beta"], 
                     test=test)
        h = self.act(h)

        return h

class Model(Chain):

    def __init__(self, device=None, act=F.relu):
        super(Model, self).__init__(
            convunit=ConvUnit(3, 64, k=3, s=1, p=1, act=act),
            resconvunit0=ResConvUnit(64, 64),
            resconvunit1=ResConvUnit(64, 64),
            resconvunit2=ResConvUnit(64, 64),
            resconvunit3=ResConvUnit(64, 64),
            resconvunit4=ResConvUnit(64, 64),
            resconvunit5=ResConvUnit(64, 64),
            linear=Linear(64*8*8, 10),
        )
        self.act = act

    def __call__(self, x, model_params, test=False):
        # Initial convolution
        mp_filtered = self._filter_model_params(model_params, "convunit")
        h = self.convunit(x, mp_filtered)

        # Residual convolution
        mp_filtered = self._filter_model_params(model_params, "resconvunit0")
        h = self.resconvunit0(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "resconvunit1")
        h = self.resconvunit1(h, mp_filtered, test)
        h = F.max_pooling_2d(h, (2, 2))  # 32 -> 16
        h = F.dropout(h, train=not test)
        
        mp_filtered = self._filter_model_params(model_params, "resconvunit2")
        h = self.resconvunit2(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "resconvunit3")
        h = self.resconvunit3(h, mp_filtered, test)
        h = F.max_pooling_2d(h, (2, 2))  # 16 -> 8
        h = F.dropout(h, train=not test)

        mp_filtered = self._filter_model_params(model_params, "resconvunit4")
        h = self.resconvunit4(h, mp_filtered, test)
        mp_filtered = self._filter_model_params(model_params, "resconvunit5")
        h = self.resconvunit5(h, mp_filtered, test)
        
        # Linear
        mp_filtered = self._filter_model_params(model_params, "linear")
        y = self.linear(h, mp_filtered)

        return y

    def _filter_model_params(self, model_params, name=""):
        """Filter parameter

        Result is parameters whose prefix of the ogirinal name is deleted, 
        the following name remains.
        """
        model_params_filtered = OrderedDict()
        for k, v in model_params.items():
            if not v.find(name):
                continue
            name = k.split("/")[1:]
            name = "/" + "/".join(name)
            model_params_filtered[name] = v
            
        return model_params_filtered
