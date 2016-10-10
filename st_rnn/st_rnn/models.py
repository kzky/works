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
        if h_t_1 == None:
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

class ElmanRNN(Chain):
    """
    One-step of ElmanNet, used with Elman

    Parameters
    -----------------
    dims: list
        each element represents dimension of a linear layer
    """
    
    def __init__(self, dims=[784, 1000, 250, 10]):
        layers = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]
            elman = Elman(d_in, d_out)
            l_name = "elman-{:03}".format(l)
            layers[l_name] = elman

        super(ElmanRNN, self).__init__(**layers)
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
