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
        h_t0 = self.h
        if h_t0 is None:
            h_t1 = self.xh(x)
        else:
            h_t1 = self.hh(h_t0) + self.xh(x)
        self.h = h_t1
        return self.h
        
    def set_state(self, h):
        """
        Parameters
        -----------------
        """
        self.h = h

    def reset_state(self, ):
        self.h = None

class ElmanOnestep(Chain):
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

        super(ElmanOnestep, self).__init__(**layers)
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

class ElmanNet(Chain):
    """
    ElmanOnestep over time.

    Parameters
    -----------------
    dims: list
        Each element represents dimension of a linear layer
    T: int
        Time length over time, i.e., the number of unroll step.
    """

    #TODO: Can we set onestep net as a chain, and BP works well with intention?
    def __init__(self, onestep, T=5):
        super(ElmanNet, self).__init__(
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

class LabeledLoss(Chain):
    def __init__(self, ):
        super(LabeledLoss, self).__init__()
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
        self.accuracy = F.accuracy(y, t)
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
        return self.loss

class RNNLabeledLosses(Chain):
    """
    Parameters
    -----------------
    T: int
    loss: LabeledLoss 
    """
    def __init__(self, T):
        
        self.T = T
        self.losses = []
        self.accuracies = []
        
        labeled_losses = OrderedDict()
        for t in range(T):
            l_name = "labeled-loss-{:03d}".format(t)
            labeled_losses[l_name] = LabeledLoss()
        
        self.labeled_losses = labeled_losses
        
    def __call__(self, y_list, y):
        """
        Parameters
        -----------------
        y_list: list of Variables
            Varialbes as predictions
        y: Variable
            Varialbe as a label
        """
        self.losses = []
        self.accuracies = []

        # y_{t-1} is as label, y_{t} is as prediction
        for y_, loss in zip(y_list, self.labeled_losses.values()):
            l = loss(y_, y)
            self.losses.append(l)
            self.accuracies.append(loss.accuracy)

        return self.losses

class RNNUnlabeledLosses(Chain):
    """
    Parameters
    -----------------
    T: int
    loss: UnlabeledLoss 
    """
    def __init__(self, T):
        
        self.T = T
        self.ulosses = []
        self.accuracies = []
        
        unlabeled_losses = OrderedDict()
        for t in range(T-1):
            l_name = "unlabeled-loss-{:03d}".format(t)
            unlabeled_losses[l_name] = UnlabeledLoss()
        
        self.unlabeled_losses = unlabeled_losses
        
    def __call__(self, y_list):
        """
        Parameters
        -----------------
        y_list: list of Variables
        """
        self.ulosses = []
        self.accuracies = []

        # y_{t-1} is as label, y_{t} is as prediction
        for y, y_, uloss in zip(y_list[0:-1], y_list[1:], self.unlabeled_losses.values()):
            l = uloss(y_, y)
            self.ulosses.append(l)

        return self.ulosses

