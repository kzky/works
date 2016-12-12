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
            bn=L.BatchNormalization(d_out, decay=0.9,
                                    use_gamma=False, use_beta=False),
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
    def __init__(self, act=F.relu, device=None):
        super(Encoder, self).__init__(
            encnet0=EncNet((784, 1000), act, device),
            encnet1=EncNet((1000, 500), act, device),
            encnet2=EncNet((500, 250), act, device),
            encnet3=EncNet((250, 100), act, device),
        )
        self.sigma = 0.3
        self.hiddens = []
        
    def __call__(self, x, noise=False, test=False, ):
        self.hiddens = []
        h = x
        if noise:
            h = x + self.encnet0.generate_norm_noise(x)
        self.hiddens.append(h)
        h = self.encnet0(x, noise, test)
        self.hiddens.append(h)
        h = self.encnet1(h, noise, test)
        self.hiddens.append(h)
        h = self.encnet2(h, noise, test)
        self.hiddens.append(h)
        h = self.encnet3(h, noise, test)
        self.hiddens.append(h)

        return h

class Combinator(Chain):
    """Two AMLP
    """
    def __init__(self, dim, act=F.relu, device=None):
        super(Combinator, self).__init__(
            linear0=L.Linear(3*dim, 2*dim),
            linear1=L.Linear(2*dim, dim),
            )
        self.act = act
        
    def __call__(self, h_l, h_v):
        """
        h_l: Varialbe
            Varialbe of lateral
        h_v: Varialbe
            Varialbe of vertial
        """
        h = F.concat((h_l, h_v, h_l*h_v))
        h = self.act(self.linear0(h))
        h = self.act(self.linear1(h))
        return h
        
class DecNet(Chain):
    """Decoder Component
    """
    def __init__(self, dim, act=F.relu, device=None):
        if len(dim) == 1:
            d_inp, = dim
            self.top = True
            super(DecNet, self).__init__(
                bn=L.BatchNormalization(d_inp, decay=0.9,
                                        use_gamma=False, use_beta=False),
                combinator=Combinator(d_inp, act, device),
            )
        else:
            self.top = False
            d_inp, d_out = dim
            super(DecNet, self).__init__(
                linear=L.Linear(d_inp, d_out),
                bn=L.BatchNormalization(d_out, decay=0.9,
                                        use_gamma=False, use_beta=False),
                combinator=Combinator(d_out, act, device),
            )

    def __call__(self, h_l, h_v, test=False):
        """
        h_l: Varialbe
            Varialbe of lateral
        h_v: Varialbe
            Varialbe of vertial
        """
        if not self.top:
            h_v = self.linear(h_v)
        h_v = self.bn(h_v)
        h_v = self.combinator(h_l, h_v)
        return h_v
        
class Decoder(Chain):
    def __init__(self, n_cls=10, act=F.relu, device=None):
        super(Decoder, self).__init__(
            decnet0=DecNet((100+n_cls, )),
            decnet1=DecNet((100+n_cls, 250+n_cls)),
            decnet2=DecNet((250+n_cls, 500+n_cls)),
            decnet3=DecNet((500+n_cls, 1000+n_cls)),
            decnet4=DecNet((1000+n_cls, 784+n_cls)),
        )
        self.hiddens = []
        
    def __call__(self, h_v, enc_hiddens, y, test=False):
        self.hiddens = []
        L = len(enc_hiddens)
        h_v = F.concat((h_v, y))
        h_l = F.concat((enc_hiddens[L-1], y))
        h_v = self.decnet0(h_l, h_v)
        self.hiddens.append(h_v)
        h_l = F.concat((enc_hiddens[L-2], y))
        h_v = self.decnet1(h_l, h_v)
        self.hiddens.append(h_v)
        h_l = F.concat((enc_hiddens[L-3], y))
        h_v = self.decnet2(h_l, h_v)
        self.hiddens.append(h_v)
        h_l = F.concat((enc_hiddens[L-4], y))
        h_v = self.decnet3(h_l, h_v)
        self.hiddens.append(h_v)
        h_l = F.concat((enc_hiddens[L-5], y))
        h_v = self.decnet4(h_l, h_v)
        h_v = F.tanh(h_v)  # align input
        self.hiddens.append(h_v)

        return h_v

class EncDecModel(Chain):
    #TODO: necessary?
    def __init__(self, act=F.relu, device=None):
        super(EncDecModel, self).__init__(
            encoder = Encoder(act, device),
            decoder = Decoder(device)
        )

    def __call__(self, x, y, noise=False, test=False):
        pass

class BranchNet(Chain):
    """MLP Branch
    Enlarge random seeds to the two times as many dimensions as dimensions.
    """
    def __init__(self, dim, device=None):
        d_inp, d_out = dim
        super(BranchNet, self).__init__(
            linear0=L.Linear(d_inp, d_out),
            linear1=L.Linear(d_out, d_out),
            bn0=L.BatchNormalization(d_out),
            # Align domain of denoising function's input
            bn1=L.BatchNormalization(d_out, use_gamma=False, use_beta=False),
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
                top=BranchNet((rdim, 100)),
                branch0=BranchNet((rdim, 250)),
                branch1=BranchNet((rdim, 500)),
                branch2=BranchNet((rdim, 500)),
                branch3=BranchNet((rdim, 1000)),
            )
            self.decnet0 = decnet0
            self.decnet1 = decnet1
            self.decnet2 = decnet2
            self.decnet3 = decnet3
            self.decnet4 = decnet4
        else:
            super(Generator, self).__init__(
                top=BranchNet((rdim, 100)),
                branch0=BranchNet((rdim, 100)),
                branch1=BranchNet((rdim, 250)),
                branch2=BranchNet((rdim, 500)),
                branch3=BranchNet((rdim, 500)),
                branch4=BranchNet((rdim, 1000)),
                decnet0=decnet0,
                decnet1=decnet1,
                decnet2=decnet2,
                decnet3=decnet3,
                decnet4=decnet4,
            )

    def __call__(self, bs, y):
        h_v = self.top(bs)
        h_l = self.branch0(bs)
        h_l = F.concat((h_l, y))
        h_v = self.decnet0(h_l, h_v)
        h_l = self.branch1(bs)
        h_l = F.concat((h_l, y))
        h_v = self.decnet1(h_l, h_v)
        h_l = self.branch2(bs)
        h_l = F.concat((h_l, y))
        h_v = self.decnet2(h_l, h_v)
        h_l = self.branch3(bs)
        h_l = F.concat((h_l, y))
        h_v = self.decnet3(h_l, h_v)
        h_l = self.branch4(bs)
        h_l = F.concat((h_l, y))
        h_v = self.decnet4(h_l, h_v)

        return h_v

class Discriminator(Chain):
    def __init__(self, act=F.relu, device=None):
        super(Discriminator, self).__init__(
            linear0=L.Linear(784, 1000),
            linear1=L.Linear(1000, 500),
            linear2=L.Linear(500, 250),
            linear3=L.Linear(250, 100),
            linear4=L.Linear(100, 1),
        )
        self.act = act
        
    def __call__(self, x):
        h = self.linear0(x)
        h = self.linear1(self.act(h))
        h = self.linear2(self.act(h))
        h = self.linear3(self.act(h)) 
        h = self.linear4(self.act(h)) 
        
        return F.sigmoid(h)

class ReconstructionLoss(Chain):
    def __init__(self, ):
        pass

    def __call__(self, x, y):
        d = np.prod(x.shape[1:])
        l = F.mean_squared_error(x, y) / d
        return l

class GanLoss(Chain):
    def __init__(self, device=None):
        pass

    def __call__(self, d_gen, d=None):
        if x:
            return F.log(d) + F.log(1 - d_gen)
        else:
            return F.log(1 - d_gen)

    
        
