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
from utils import to_device
from chainer_fix import BatchNormalization

class MLPEnc(Chain):

    def __init__(self,
                     dims,
                     act=F.relu,
                     noise=False,
                     rc=False,
                     lateral=False,
                     device=None):

        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_in, d_out)
            l_name = "linear-enc-{:03}".format(l)
            linears[l_name] = linear

            # Normalization and BatchCorrection
            batch_norm = L.BatchNormalization(d_out, decay=0.9)
            bn_name = "bn-enc-{:03d}".format(l)
            batch_norms[bn_name] = batch_norm

        layers.update(linears)
        layers.update(batch_norms)
        
        super(MLPEnc, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = batch_norms
        self.act = act
        self.noise = noise
        self.rc = rc
        self.lateral = lateral
        self.device = device
        self.hiddens = []

    def __call__(self, x, test):
        h = x
        self.hiddens = []
        for i, layers in enumerate(zip(
            self.linears.values(), self.batch_norms.values())):

            linear, batch_norm = layers

            # Add noise
            if self.noise and not test:
                if np.random.randint(0, 2):
                    n = np.random.normal(0, 0.03, h.data.shape).astype(np.float32)
                    n_ = Variable(to_device(n, self.device))
                    h = h + n_

            # Linear
            h = linear(h)

            # Batchnorm
            h = batch_norm(h, test)

            # Activation
            h = self.act(h)

            # RC after non-linearity
            if self.rc and i != len(self.dims) - 2:
                self.hiddens.append(h)

        return h

    def _add_noise(self, h, test):
        if self.noise and not test:
            if np.random.randint(0, 2):
                n = np.random.normal(0, 0.03, h.data.shape).astype(np.float32)
                n_ = Variable(to_device(n, self.device))
                h = h + n_
        return h
        

class Denoise(Chain):
    def __init__(self, dim):
        super(Denoise, self).__init__(
            a0=L.Scale(W_shape=(dim, )),
            a1=L.Scale(W_shape=(dim, )),
            a2=L.Scale(W_shape=(dim, )),
            a3=L.Bias(shape=(dim, )),
            a4=L.Scale(W_shape=(dim, )),
            b0=L.Scale(W_shape=(dim, )),
            b1=L.Scale(W_shape=(dim, )),
            b2=L.Scale(W_shape=(dim, )),
            b3=L.Bias(shape=(dim, )),
            )

    def __call__(self, x, y):
        xy = x * y
        a = self.a3(self.a0(x) + self.a1(y) + self.a2(xy))
        b = self.b3(self.b0(x) + self.b1(y) + self.b2(xy))
        
        return b + self.a4(F.sigmoid(a))

class MLPDec(Chain):

    def __init__(self, dims, act=F.relu,
                     rc=False,
                     lateral=False,
                     mlp_enc=None,
                     device=None):
        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        denoises = OrderedDict()
        
        dims_reverse = dims[::-1]
        for l, d in enumerate(zip(dims_reverse[0:-1], dims_reverse[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_in, d_out)
            l_name = "linear-dec-{:03}".format(l)
            linears[l_name] = linear

            # Normalization and BatchCorrection
            batch_norm = BatchNormalization(d_out, decay=0.9,
                                            use_gamma=False, use_beta=False, device=device)
            bn_name = "bn-dec-{:03d}".format(l)
            batch_norms[bn_name] = batch_norm

            # Denoise
            if lateral and l != 0:
                dn_name = "dn-dec-{:03d}".format(l)
                denoises[dn_name] = Denoise(d_in)
                                
        layers.update(linears)
        layers.update(batch_norms)
        layers.update(denoises) if lateral else None
        
        super(MLPDec, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = batch_norms
        self.denoises = denoises
        self.act = act
        self.rc = rc
        self.lateral = lateral
        self.device = device
        self.hiddens = []
        self.mlp_enc = mlp_enc
            
    def __call__(self, x, test):
        h = x
        self.hiddens = []
        for i, layers in enumerate(zip(
            self.linears.values(), self.batch_norms.values())):
            linear, batch_norm = layers

            # Linear
            h = linear(h)

            # Batchnorm
            h = batch_norm(h, test)

            # Activation, no need for non-linearity for RC of x
            if not self.lateral != len(self.dims) - 2:
                h = self.act(h)

            # Denoise
            if self.lateral and i != len(self.dims) - 2:
                denoise = self.denoises.values()[i]
                h_enc = self.mlp_enc.hiddens[::-1][i]
                h = denoise(h_enc, h)

            # RC after non-linearity
            if self.rc and i != len(self.dims) - 2:
                self.hiddens.append(h)
                
        return h
            
class SupervizedLoss(Chain):

    def __init__(self, ):
        super(SupervizedLoss, self).__init__()
        self.loss = None
        
    def __call__(self, y, t):
        self.loss = F.softmax_cross_entropy(y, t)
        return self.loss

class ReconstructionLoss(Chain):

    def __init__(self,
                     noise=False,
                     rc=False,
                     lateral=False):

        super(ReconstructionLoss, self).__init__()
        self.noise = noise
        self.rc = rc
        self.lateral = lateral
        self.loss = None
        
    def __call__(self, x_recon, x, enc_hiddens, dec_hiddens):
        """
        Parameters
        -----------------
        x_recon: Variable to be reconstructed as label
        x: Variable to be reconstructed as label
        enc_hiddens: list of Variable
        dec_hiddens: list of Varialbe
        """

        # Lateral Recon Loss
        recon_loss = 0
        if self.rc and enc_hiddens is not None:
            for h0, h1 in zip(enc_hiddens[::-1], dec_hiddens):
                d = np.prod(h0.data.shape[1:])
                recon_loss += F.mean_squared_error(h0, h1) / d

        # Reconstruction Loss
        if x_recon is not None:
            d = np.prod(x.data.shape[1:])
            recon_loss += F.mean_squared_error(x_recon, x) / d

        self.loss = recon_loss
        
        return self.loss

class PseudoSupervisedLoss(Chain):
    """Compute cross entropy between y_t and y_{t+1}.
    """

    def __init__(self, ):
        pass

    def __call__(self, y, t):
        t_normalized = F.softmax(t)
        y_log_softmax = F.log_softmax(y)
        n = y.data.shape[0]

        return - F.sum(t_normalized * y_log_softmax) / n

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

class MLPEncDecModel(Chain):
    def __init__(self,
                     dims,
                     act=F.relu,
                     noise=False,
                     rc=False,
                     lateral=False,
                     device=None):
        # Constrcut models
        mlp_enc = MLPEnc(
            dims=dims,
            act=act,
            noise=noise,
            rc=rc,
            lateral=lateral,
            device=device)
        mlp_dec = MLPDec(
            dims=dims,
            act=act,
            rc=rc,
            lateral=lateral,
            mlp_enc=mlp_enc,
            device=device)
        self.supervised_loss = SupervizedLoss()
        self.recon_loss = ReconstructionLoss(noise, rc, lateral)
        self.pseudo_supervised_loss = PseudoSupervisedLoss()
        self.kl_loss = KLLoss()

        super(MLPEncDecModel, self).__init__(
            mlp_enc=mlp_enc,
            mlp_dec=mlp_dec)

    def __call__(self, x_l, y_l, x_u, y_u):
        pass
