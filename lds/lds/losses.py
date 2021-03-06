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

class ReconstructionLoss(Chain):

    def __init__(self,
                     ):
        super(ReconstructionLoss, self).__init__()
        self.loss = None
        
    def __call__(self, x_recon, x):
        bs = x.shape[0]
        d = np.prod(x.shape[1:])
        self.loss = F.mean_squared_error(x_recon, x) / d

        return self.loss

class ReconstructionLoss1(Chain):

    def __init__(self,
                     ):
        super(ReconstructionLoss1, self).__init__()
        self.loss = None
        
    def __call__(self, x_recon, x):
        bs = x.shape[0]
        d = np.prod(x.shape[1:])
        self.loss = F.mean_absolute_error(x_recon, x) / d

        return self.loss

class NegativeEntropyLoss(Chain):

    def __init__(self, test=False):
        super(NegativeEntropyLoss, self).__init__()
        self.loss = None
        
    def __call__(self, y, ):
        bs = y.data.shape[0]
        d = np.prod(y.data.shape[1:])

        y_normalized = F.softmax(y)
        y_log_softmax = F.log_softmax(y)
        self.loss = - F.sum(y_normalized * y_log_softmax) / bs / d

        return self.loss

class EntropyLossForAll(Chain):

    def __init__(self, test=False):
        super(EntropyLossForAll, self).__init__()
        self.loss = None
        
    def __call__(self, y, ):
        bs = y.data.shape[0]
        d = np.prod(y.data.shape[1:])

        y = F.reshape(y, (bs, d))

        y_normalized = F.softmax(y)
        y_log_softmax = F.log_softmax(y)
        self.loss = - F.sum(y_normalized * y_log_softmax) / bs / d

        return self.loss

class EntropyLossForEachMap(Chain):

    def __init__(self, test=False):
        super(EntropyLossForEachMap, self).__init__()
        self.loss = None
        
    def __call__(self, y, ):
        bs = y.data.shape[0]
        d = np.prod(y.data.shape[1])
        if len(y.shape) > 2:
            s = np.prod(y.data.shape[2:])

            y = F.reshape(y, (bs, d, s))
            y = F.transpose(y, (0, 2, 1))

            y_normalized = F.softmax(y, use_cudnn=False)
            y_log_softmax = F.log_softmax(y, use_cudnn=False)
            self.loss = - F.sum(y_normalized * y_log_softmax) / bs / s
        else:
            y_normalized = F.softmax(y)
            y_log_softmax = F.log_softmax(y)
            self.loss = - F.sum(y_normalized * y_log_softmax) / bs / d

        return self.loss
    
class NegativeEntropyHingeLoss(Chain):

    def __init__(self, test=False):
        super(NegativeEntropyHingeLoss, self).__init__()
        self.loss = None
        
    def __call__(self, y, ):
        bs = y.data.shape[0]
        d = np.prod(y.data.shape[1:])

        y_normalized = F.softmax(y)
        y_log_softmax = F.log_softmax(y)
        negentropy = F.sum(y_normalized * y_log_softmax, axis=1) / d

        #zeros = to_device(np.zeros(bs).astype(np.float32), 2)
        ones = to_device(-np.ones(bs).astype(np.float32), 2)
        self.loss = F.sum(F.maximum(
            Variable(ones), 
            - negentropy)) / bs
        
        return self.loss

class KLLoss(Chain):
    """Compute cross entropy between y_t and y_{t+1}.
    """

    def __init__(self, ):
        pass
        
    def __call__(self, y, t):
        t_normalized = F.softmax(t)
        t_log_softmax = F.log_softmax(t)
        y_log_softmax = F.log_softmax(y)
        n = y.data.shape[0]

        return F.sum((t_normalized * t_log_softmax) \
                         - (t_normalized * y_log_softmax)) / n
    
class JensenShannonDivergenceLoss(Chain):

    def __init__(self, test=False):
        super(JensenShannonDivergenceLoss, self).__init__()

    def __call__(self, y0, y1):
        bs = y0.data.shape[0]
        d = np.prod(y0.data.shape[1:])

        y0_softmax = F.softmax(y0)
        y1_softmax = F.softmax(y1)

        y0_log_softmax = F.log_softmax(y0)
        y1_log_softmax = F.log_softmax(y1)

        kl0 = F.sum(y0_softmax * (y0_log_softmax - y1_log_softmax)) / bs / d
        kl1 = F.sum(y1_softmax * (y1_log_softmax - y0_log_softmax)) / bs / d

        return (kl0 + kl1) / 2

class FrobeniousConvLoss(object):

    def __init__(self, device=None):
        super(FrobeniousConvLoss, self).__init__()
        self.device = device
        
    def __call__(self, h):
        if len(h.shape) != 4:
            return 0
        
        # (b, c, h, w) -> (b, h, w, c) -> (b, h*w, c)
        h = F.transpose(h, (0, 2, 3, 1))
        shape = h.shape
        b, n, c =  shape[0], shape[1]*shape[2], shape[3]
        h = F.reshape(h, (b, n, c))
        s = 0
        xp = cuda.get_array_module(h.data)
        I_ = xp.identity(n)
        I_ = Variable(to_device(I_, device))
        for h_ in h:
            s += F.sum(F.square(F.linear(h_, h_) - I_))
        l = s / (b * n * c)
        return l
        
        
        
        

        
        
    
