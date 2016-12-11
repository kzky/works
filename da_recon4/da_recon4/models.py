"""Models
"""
import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer.cuda import cupy as cp
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from utils import to_device

#TODO: wscale

class EncNet(Chain):
    """Encoder Component
    """
    def __init__(self, dim, act=F.relu, device=None, ):
        d_inp, d_out = dim
        super(EncNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=L.Batchnorm(d_out, decay=0.9, use_gamm=False, use_beta=False),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )
        self.sigma = 0.3
        self.act = act

    def __call__(self, h, noise=False, test=False):
        h = self.linear(h)
        h = self.bn(h, test)
        if noise:
            h = h + self.generate_norm_noise(h)
        h = self.sb(h)
        h = self.act(h)
        return h

    def generate_norm_noise(self, h):
        bs, d = h.shape
        if self.device:
            r = Variable(
                cuda.to_gpu(cp.random.randn(bs, d).astype(cp.float32), self.device))
            return r * self.sigma
        else:
            return Variable(np.random.randn(bs, d).astype(np.float32)) * self.sigma

class Encoder(Chain):
    def __init__(self, device=None):
        super(Encoder, self).__init__(
            encnet0=EncNet((780, 1000)),
            encnet1=EncNet((1000, 500)),
            encnet2=EncNet((500, 250)),
            encnet3=EncNet((250, 100)),
        )
        self.sigma = 0.3
        self.hiddens = []
        
    def __call__(self, x, noise=False, test=False, ):
        self.hiddens = []

        if noise:
            x = x + self.encnet0.generate_norm_noise(x)
        self.hiddens.append(h)
        h = encnet0(x, noise, test)
        self.hiddens.append(h)
        h = encnet1(h, noise, test)
        self.hiddens.append(h)
        h = encnet2(h, noise, test)
        self.hiddens.append(h)
        h = encnet3(h, noise, test)
        self.hiddens.append(h)

        return h

class Denoise(Chain):
    def __init__(self, dim, device=None):
        #TODO: Initialization
        super(Denoise, self).__init__(
            a0=L.Scale(W_shape=(dim, )),
            a1=L.Scale(W_shape=(dim, )),
            a2=L.Scale(W_shape=(dim, )),
            a3=L.Bias(shape=(dim, )),
            a4=L.Scale(shape=(dim, )),
            b0=L.Scale(W_shape=(dim, )),
            b1=L.Scale(W_shape=(dim, )),
            b2=L.Scale(W_shape=(dim, )),
            b3=L.Bias(shape=(dim, )),
            )

    def __call__(self, x, y):
        """
        x: Varialbe
            Varialbe of lateral
        y: Varialbe
            Varialbe of vertial
        """
        xy = x * y
        a = self.a3(self.a0(x) + self.a1(y) + self.a2(xy))
        b = self.b3(self.b0(x) + self.b1(y) + self.b2(xy))
        
        return b + self.a4(F.sigmoid(a))
        
class DecNet(Chain):
    """Decoder Component
    """
    def __init__(self, dim, device=None):
        d_inp, d_out = dim
        super(DecNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=L.Batchnorm(d_out, decay=0.9, use_gamm=False, use_beta=False),
            denoise=Denoise(d_out),
        )

    def __call__(self, z, h, test=False):
        h = self.linear(h)
        h = self.bn(h)
        h = self.denoise(z, h)
        return h
        
class Decoder(Chain):
    def __init__(self, device=None):
        super(Decoder, self).__init__(
            bn=L.Batchnorm(d_out, decay=0.9, use_gamm=False, use_beta=False),
            decnet0=DecNet((100, 100)),
            decnet1=DecNet((100, 250)),
            decnet2=DecNet((250, 500)),
            decnet3=DecNet((500, 1000)),
            decnet4=DecNet((1000, 780)),
        )
        self.hiddens = []
        
    def __call__(self, h, enc_hiddens, test=False):
        self.hiddens = []
        L = len(enc_hiddens)
        
        h = self.bn(h, test)
        h = decnet0(enc_hiddens[L-1], h)
        self.hiddens.apppend(h)
        h = decnet1(enc_hiddens[L-2], h)
        self.hiddens.apppend(h)
        h = decnet2(enc_hiddens[L-3], h)
        self.hiddens.apppend(h)
        h = decnet3(enc_hiddens[L-4], h)
        self.hiddens.apppend(h)
        h = decnet4(enc_hiddens[L-5], h)
        h = F.tanh(h)  # align input
        self.hiddens.apppend(h)

        return h

class MLPBranch(Chain):
    """MLP Branch
    Enlarge random seeds to the two times as many dimensions as dimensions.
    """
    def __init__(self, dim, fix=False, device=None):
        d_inp, d_out = dim
        super(MLPBranch, self).__init__(
            linear0=L.Linear(d_inp, d_out),
            linear1=L.Linear(d_out, d_out),
            bn0=L.Batchnorm(d_out),
            # Align domain of denoising function's input
            bn1=L.Batchnorm(d_out, use_gamma=False, use_beta=False),
        )
        self.dim = d_inp
        
    def __call__(self, bs, ):
        u = self.generate_unif(bs, self.dim)
        h = self.linear0(u)
        h = self.bn0(h)
        h = self.linear1(h)
        h = self.bn1(h)
        return h
                        
    def generate_unif(self, bs, dim=100):
        if self.device:
            r = Variable(cuda.to_gpu(
                cp.random.uniform(-1, 1, (bs, dim)).astype(cp.float32), self.device))
            return r
        else:
            return Variable(np.random.uniform(-1, 1, (bs, dim)).astype(np.float32))

class Generator(Chain):

    def __init__(self, decoder, fix=False, device=None):
        decnet0 = decoder.decnet0 
        decnet1 = decoder.decnet1 
        decnet2 = decoder.decnet2 
        decnet3 = decoder.decnet3
        decnet4 = decoder.decnet4
        rdim = 100
        if fix:
            super(Generator, self).__init__(
                top=MLPBranch((rdim, 100)),
                branch0=MLPBranch((rdim, 250)),
                branch1=MLPBranch((rdim, 500)),
                branch2=MLPBranch((rdim, 500)),
                branch3=MLPBranch((rdim, 1000)),
            )
            self.decnet0 = decnet0
            self.decnet1 = decnet1
            self.decnet2 = decnet2
            self.decnet3 = decnet3
            self.decnet4 = decnet4
        else:
            super(Generator, self).__init__(
                top=MLPBranch((rdim, 100)),
                branch0=MLPBranch((rdim, 100)),
                branch1=MLPBranch((rdim, 250)),
                branch2=MLPBranch((rdim, 500)),
                branch3=MLPBranch((rdim, 500)),
                branch4=MLPBranch((rdim, 1000)),
                decnet0=decnet0,
                decnet1=decnet1,
                decnet2=decnet2,
                decnet3=decnet3,
                decnet4=decnet4,
            )

    def __call__(self, bs):
        h_v = self.top(bs)
        h_l = self.branch0(bs)
        h_v = self.decnet0(h_l, h_t)
        h_l = self.branch1(bs)
        h_v = self.decnet1(h_l, h_v)
        h_l = self.branch2(bs)
        h_v = self.decnet2(h_l, h_v)
        h_l = self.branch3(bs)
        h_v = self.decnet3(h_l, h_v)
        h_l = self.branch4(bs)
        h_v = self.decnet4(h_l, h_v)

        return h_v

class Discriminator(Chain):

    def __init__(self, device=None):
        pass

    def __call__(self, ):
        pass


class Reconstruction(Chain):
    def __init__(self, ):
        pass

    def __call__(self, x, y):
        d = np.prod(x.shape[1:])
        l = F.mean_squared_error(x, y) / d
        return l
        
        
