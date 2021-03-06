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
                     device=None):

        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        scale_biases = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_in, d_out, wscale=1/np.sqrt(d_in))
            l_name = "linear-enc-{:03}".format(l)
            linears[l_name] = linear

            # Normalization and BatchCorrection
            batch_norm = BatchNormalization(d_out, decay=0.9, 
                                            use_gamma=False, use_beta=False)
            bn_name = "bn-enc-{:03d}".format(l)
            batch_norms[bn_name] = batch_norm

            scale_bias = L.Scale(W_shape=d_out, bias_term=True)
            sb_name = "sb-enc-{:03d}".format(l)
            scale_biases[sb_name] = scale_bias

        layers.update(linears)
        layers.update(batch_norms)
        layers.update(scale_biases)
        
        super(MLPEnc, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = batch_norms
        self.scale_biases = scale_biases
        self.act = act
        self.noise = noise
        self.rc = rc
        self.device = device
        self.hiddens = []

    def __call__(self, x, test):
        h = x
        self.hiddens = []
        for i, layers in enumerate(zip(
                self.linears.values(), self.batch_norms.values(), self.scale_biases.values())):

            linear, batch_norm, scale_bias = layers

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

            # Scale bias
            h = scale_bias(h)            

            # Activation  #TODO: have to add in the last layer?
            h = self.act(h)

            # RC after non-linearity
            if self.rc and i != len(self.dims) - 2:
                self.hiddens.append(h)

        return h

class MLPDec(Chain):

    def __init__(self, dims, act=F.relu,
                     rc=False,
                     mlp_enc=None,
                     device=None):
        # Setup layers
        layers = {}
        linears = OrderedDict()
        batch_norms = OrderedDict()
        scale_biases = OrderedDict()
        
        dims_reverse = dims[::-1]
        for l, d in enumerate(zip(dims_reverse[0:-1], dims_reverse[1:])):
            d_in, d_out = d[0], d[1]

            # Linear
            linear = L.Linear(d_in, d_out, wscale=1/np.sqrt(d_in))
            l_name = "linear-dec-{:03}".format(l)
            linears[l_name] = linear

            # Normalization and BatchCorrection
            batch_norm = BatchNormalization(d_out, decay=0.9, 
                                            use_gamma=False, use_beta=False)
            bn_name = "bn-dec-{:03d}".format(l)
            batch_norms[bn_name] = batch_norm

            scale_bias = L.Scale(W_shape=d_out, bias_term=True)
            sb_name = "sb-enc-{:03d}".format(l)
            scale_biases[sb_name] = scale_bias

        layers.update(linears)
        layers.update(batch_norms)
        layers.update(scale_biases)
        
        super(MLPDec, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.linears = linears
        self.batch_norms = batch_norms
        self.scale_biases = scale_biases
        self.act = act
        self.rc = rc
        self.device = device
        self.hiddens = []
        self.mlp_enc = mlp_enc
            
    def __call__(self, x, test):
        h = x
        self.hiddens = []
        for i, layers in enumerate(zip(
                self.linears.values(), self.batch_norms.values(), self.scale_biases.values())):
            linear, batch_norm, scale_bias = layers

            # Linear
            h = linear(h)

            # Batchnorm
            h = batch_norm(h, test)

            # Scale bias
            h = scale_bias(h)            

            # Activation, no need for non-linearity for RC of x
            if i != len(self.dims) - 2:
                h = self.act(h)

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
                     ):

        super(ReconstructionLoss, self).__init__()
        self.noise = noise
        self.rc = rc
        self.loss = None
        
    def __call__(self, x_recon, x, enc_hiddens, dec_hiddens, scale=True):
        """
        Parameters
        -----------------
        x_recon: Variable to be reconstructed as label
        x: Variable to be reconstructed as label
        enc_hiddens: list of Variable
        dec_hiddens: list of Varialbe
        """
        recon_loss = 0
        
        # Lateral Recon Loss
        if self.rc and enc_hiddens is not None:
            for h0, h1 in zip(enc_hiddens[::-1], dec_hiddens):
                l = F.mean_squared_error(h0, h1)
                if scale:
                    d = np.prod(h0.data.shape[1:])
                    l = l / d
                recon_loss += l
                
        # Reconstruction Loss
        if x_recon is not None:
            l = F.mean_squared_error(x_recon, x)
            if scale:
                d = np.prod(x.data.shape[1:])
                l = l / d
            recon_loss +=  l

        self.loss = recon_loss
        return self.loss

class KLReconstructionLoss(Chain):

    def __init__(self,
                     rc=False,
                     ):

        super(KLReconstructionLoss, self).__init__()
        self.rc = rc
        self.loss = None
        
    def __call__(self, x_recon, x, enc_hiddens, dec_hiddens, scale=True):
        """
        Parameters
        -----------------
        x_recon: Variable to be reconstructed as label
        x: Variable to be reconstructed as label
        enc_hiddens: list of Variable
        dec_hiddens: list of Varialbe
        """
        kl_recon_loss = 0
        
        # Lateral Recon Loss
        if self.rc and enc_hiddens is not None:
            for h0, h1 in zip(enc_hiddens[::-1], dec_hiddens):
                n = h0.shape[0]
                d = np.prod(h0.shape[1:])
                p = F.softmax(h0)
                log_p = F.log_softmax(h0)
                log_q = F.log_softmax(h1)
                l = F.sum(p * (log_p - log_q)) / n / d
                kl_recon_loss += l

        self.loss = kl_recon_loss
        return self.loss

class CrossCovarianceLoss(Chain):

    def __init__(self,
                     rc=False,
                     ):

        super(CrossCovarianceLoss, self).__init__()
        self.rc = rc
        self.loss = None
        
    def __call__(self, x_recon, x, enc_hiddens, dec_hiddens, scale=True):
        """
        Parameters
        -----------------
        x_recon: Variable to be reconstructed as label
        x: Variable to be reconstructed as label
        enc_hiddens: list of Variable
        dec_hiddens: list of Varialbe
        """
        cc_loss = 0
        
        # Lateral Recon Loss
        if self.rc and enc_hiddens is not None:
            for h0, h1 in zip(enc_hiddens[::-1], dec_hiddens):
                n = h0.shape[0]
                d = np.prod(h0.shape[1:])
                l = F.cross_covariance(h0, h1) / n / d
                cc_loss += l

        self.loss = cc_loss
        return self.loss
    
class NegativeEntropyLoss(Chain):
    """Compute cross entropy between y_t and y_{t+1}.
    """

    def __init__(self, test=False):
        self.test = test


    def __call__(self, y, hiddens=None, scale=True):
        ne_loss = 0
        
        # NE for hiddens
        if hiddens is not None:
            for h in hiddens:
                h_normalized = F.softmax(h)
                h_log_softmax = F.log_softmax(h)
                n = h.data.shape[0]
                l = - F.sum(h_normalized * h_log_softmax) / n 
                if scale:
                    d = np.prod(h.data.shape[1:])
                    l = l / d
                ne_loss += l
                
        # NE for output
        y_normalized = F.softmax(y)
        y_log_softmax = F.log_softmax(y)
        n = y.data.shape[0]
        l = - F.sum(y_normalized * y_log_softmax) / n 
        if scale:
            d = np.prod(y.data.shape[1:])
            l = l / d
        ne_loss += l
        return ne_loss

class GraphLoss(Chain):
    def __call__(self, x, hiddens, y):
        # Input Similarity
        d = np.prod(x.shape[1:])
        distmat = self._calc_distmat(x)
        sim_input = F.exp(- distmat) / d
        
        # Feature Similarity
        sim_feats = 0
        for h in hiddens:
            d = np.prod(h.shape[1:])
            distmat = self._calc_distmat(h)
            sim_feats += F.exp(- distmat) / d

        # Label Similarity
        d = np.prod(y.shape[1:])
        sim_label = self._calc_distmat(y) / d

        # Graph Loss
        bs_2 = y.shape[0] ** 2
        loss = F.sum((sim_input + sim_feats) * sim_label) / bs_2
        return loss

    def _calc_distmat(self, h):
        bs = h.shape[0]
        
        h_l2_2 = F.sum(h**2, axis=1)
        H = F.broadcast_to(h_l2_2, (bs, bs))
        H_t = F.transpose(H)
        XX = F.linear(h, h)

        return (H_t - 2*XX + H)

class MLPEncDecModel(Chain):
    def __init__(self,
                     dims,
                     act=F.relu,
                     noise=False,
                     rc=False,
                     device=None):
        # Constrcut models
        mlp_enc = MLPEnc(
            dims=dims,
            act=act,
            noise=noise,
            rc=rc,
            device=device)
        mlp_dec = MLPDec(
            dims=dims,
            act=act,
            rc=rc,
            mlp_enc=mlp_enc,
            device=device)
        self.supervised_loss = SupervizedLoss()
        self.recon_loss = ReconstructionLoss(noise, rc)
        self.neg_ent_loss = NegativeEntropyLoss()
        self.graph_loss = GraphLoss()
        self.kl_recon_loss = KLReconstructionLoss(rc)
        self.cc_loss = CrossCovarianceLoss(rc)

        super(MLPEncDecModel, self).__init__(
            mlp_enc=mlp_enc,
            mlp_dec=mlp_dec)

    def __call__(self, x_l, y_l, x_u, y_u):
        pass
