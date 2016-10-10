"""Models
"""
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time

class Elman(Chain):
    """
    Parameters
    -----------------
    x_in: int
    h_out: int
    """
    def __init__(self, x_in, h_out):
        super(Elman, self).__init__(
            xh=L.Linear(x_in, h_out),
            hh=L.Linear(h_out, h_out),
        )

        self.h = None
        
    def __call__(self, x):
        """One-step forward
        Parameters
        -----------------
        x: Variable
            input from the previous layer, i.e., the bottom layer of one-step RNN
        """
        h_t_1 = self.h
        if h_t_1 is None:
            h_t = self.xh(x)
        else:
            h_t = self.hh(h_t_1) + self.xh(x)
        self.h = h_t
        return self.h
        
    def set_state(self, h):
        """
        Parameters
        -----------------
        """
        self.h = h

    def reset_state(self, ):
        self.h = None

class ElmanNet(Chain):
    """
    One-step of ElmanRNN, used with Elman

    Parameters
    -----------------
    dims: list
        Each element represents dimension of a linear layer
    """
    
    def __init__(self, dims):
        layers = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]
            elman = Elman(d_in, d_out)
            l_name = "elman-{:03}".format(l)
            layers[l_name] = elman

        super(ElmanNet, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
            
    def __call__(self, x):
        """
        Parameters
        -----------------
        x: Variable
            Input variable
        """
        h = x
        for elman in self.layers.values():
            h = elman(h)

        return h

    def set_states(self,  hiddens):
        """Set all states.
        
        Parameters
        -----------------
        hiddens: list of Variables
        """
        if len(hiddens) != len(self.layers):
            raise ValueError("Length differs between hiddens and self.layers")
            
        for elman, h in zip(self.layers.values(), hiddens):
            elman.set_state(h)

    def reset_states(self,):
        """Reset all states.
        """
        for elman in self.layers.values():
            elman.reset_state()

    def get_states(self, ):
        """Get all states
        """
        hiddens = []
        for elman in self.layers.values():
            hiddens.append(elman.h)
        return hiddens

class ElmanRNN(Chain):
    """
    ElmanNet over time.

    Parameters
    -----------------
    dims: list
        Each element represents dimension of a linear layer
    T: int
        Time length over time, i.e., the number of unroll step.
    """

    #TODO: Can we set onestep net as a chain, and BP works well with intention?
    def __init__(self, onestep, T=5):
        super(ElmanRNN, self).__init__(
            onestep=onestep,
        )
        self.T = T

    def __call__(self, x_list):
        """
        Parameters
        -----------------
        x_list: list of Variables
            Input variables over time
        """
        y_list = []
        for t in range(self.T):
            y = self.onestep(x_list[t])
            y_list.append(y)
        
        return y_list

class LabledLoss(Chain):
    def __init__(self, ):
        super(LabledLoss, self).__init__()
        self.loss = None
        self.accuracy = None
        self.pred = None
        
    def __call__(self, y, t):
        """
        y: Variable
            Prediction Variable of shape (bs, cls)
        t: Variable
            Label
        """
        self.loss = F.softmax_cross_entropy(y, t)
        self.accauracy = F.accuracy(y, t)
        return self.loss

class UnlabeledLoss(Chain):
    def __init__(self, ):
        super(UnlabeledLoss, self).__init__()
        self.loss = None
        self.accuracy = None
        self.pred = None
        
    def __call__(self, y_,  y):
        """
        y_: Variable
            Prediction Variable of shape (bs, cls)
        y: Variable
            Prediction Variable of shape (bs, cls) as label
        """
        self.pred_ = F.softmax(y_)
        self.pred = F.softmax(y)
        self.loss = - F.sum(self.pred * F.log(self.pred_)) / len(y_)
        self.accauracy = F.accuracy(y, t)
        return self.loss

class RNNLosses(Chain):
    def __init__(self, T):
        
        self.T = T
        self.losses = []
        self.accuracies = []
        
        unlabeled_losses = OrderedDict()
        for t in range(T-1):
            l_name = "{unlabeled-loss-:03d}".format(t)
            unlabeled_losses[l_name] = UnlabeledLoss()
        
        #super(Loss, self).__init__(**unlabeled_losses)
        self.unlabeled_losses = unlabeled_losses
        
    def __call__(self, y_list):
        self.losses = []
        self.accuracies = []

        # y_{t-1} is as label, y_{t} is as prediction
        for y, y_, uloss in zip(y_list[0:-1], y_list[1:], self.unlabeled_losses.values()):
            l = uloss(y_, y)
            self.losses.append(l)
            self.accuracies.append(l.accuracy)

        return self.losses

#class ElmanRNNModel(Chain):
#    """Very Plain ElmanRNN Model
#    """
#    def __init__(self, dims, T, T0):
#        self.dims = dims
#        self.T = T
#        self.T0 = T0
#        onestep = ElmanNet(dims)
#        elman_rnn = ElmanRNN(onestep, T)
#        loss = LabledLoss()
#        rnn_losses = RNNLosses(T)
#        super(ElmanRNNModel, self).__init__(
#            elman_rnn=elman_rnn
#        )
        
        
def forward_with_elman_rnn(onestep, elman_rnn, loss, rnn_losses):
    pass
