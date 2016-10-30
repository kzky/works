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

"""
Elman
"""
class Elman(Chain):
    """
    Parameters
    -----------------
    x_in: int
    h_out: int
    """
    def __init__(self, x_in, h_out, act):
        super(Elman, self).__init__(
            xh=L.Linear(x_in, h_out),
            hh=L.Linear(h_out, h_out),
        )

        self.h = None
        self.act = act
        
    def __call__(self, x):
        """One-step forward
        Parameters
        -----------------
        x: Variable or None
            input from the previous layer, i.e., the bottom layer of one-step RNN
        """
        h_t0 = self.h
        if x is None:
            h_t1 = self.hh(h_t0)
        elif h_t0 is None:
            h_t1 = self.xh(x)
        else:
            h_t1 = self.hh(h_t0) + self.xh(x)
        self.h = self.act(h_t1)
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
    
    def __init__(self, dims, act=F.relu):
        layers = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]
            elman = Elman(d_in, d_out, act)
            l_name = "elman-{:03}".format(l)
            layers[l_name] = elman

        super(ElmanOnestep, self).__init__(**layers)
        self.dims = dims
        self.layers = layers
        self.act = act
            
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

"""
LSTM
"""
class LSTM(L.LSTM):
    def __init__(self, in_size, out_size, **kwargs):
        super(LSTM, self).__init__(in_size, out_size, **kwargs)

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if self.upward.has_uninitialized_params:
            in_size = x.size // len(x.data)
            self.upward._initialize_params(in_size)
            self._initialize_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than the '
                       'size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((batch, self.state_size), dtype=x.dtype),
                volatile='auto')
            # Need to pass cell to GPU!
            device = cuda.get_device(lstm_in)
            self.c.to_gpu(device)
            
        self.c, y = lstm.lstm(self.c, lstm_in)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
    
class LSTMOnestep(Chain):
    """
    One-step of LSTMRNN, used with LSTM

    Parameters
    -----------------
    dims: list
        Each element represents dimension of a linear layer
    """
    
    def __init__(self, dims):
        layers = OrderedDict()
        for l, d in enumerate(zip(dims[0:-1], dims[1:])):
            d_in, d_out = d[0], d[1]
            lstm = LSTM(d_in, d_out)
            l_name = "lstm-{:03}".format(l)
            layers[l_name] = lstm

        super(LSTMOnestep, self).__init__(**layers)
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
        for lstm in self.layers.values():
            h = lstm(h)
        return h

    def set_states(self,  states):
        """Set all states.
        
        Parameters
        -----------------
        states: list of tuple of Variables
            First element of the tuple is for cell and the second for hidden.
        """
        if len(states) != len(self.layers):
            raise ValueError("Length differs between states and `layers`")
            
        for lstm, state in zip(self.layers.values(), states):
            lstm.set_state(state[0], state[1])

    def reset_states(self,):
        """Reset all states.
        """
        for lstm in self.layers.values():
            lstm.reset_state()

    def get_states(self, ):
        """Get all states
        """
        states = []
        for lstm in self.layers.values():
            states.append((lstm.c, lstm.h))
        return states

class LSTMNet(Chain):
    """
    LSTMOnestep over time.

    Parameters
    -----------------
    dims: list
        Each element represents dimension of a linear layer
    T: int
        Time length over time, i.e., the number of unroll step.
    """

    #TODO: Can we set onestep net as a chain, and BP works well with intention?
    def __init__(self, onestep, T=5):
        super(LSTMNet, self).__init__(
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
    """
    Parameters
    -----------------
    l_type: str
        Label type, "soft" or "hard"
    """

    def __init__(self, l_type="soft"):
        super(UnlabeledLoss, self).__init__()
        self.l_type = l_type
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
        if self.l_type == "soft": 
            self.pred_ = F.softmax(y_)
            self.pred = F.softmax(y)
            self.loss = - F.sum(self.pred * F.log(self.pred_)) / len(y_)
        elif self.l_type == "hard":
            t = F.argmax(y, axis=1)
            self.loss = F.softmax_cross_entropy(y_, t)
            
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
    l_type: str
        "soft" or "hard"
    """
    def __init__(self, T, l_type="soft"):
        
        self.T = T
        self.l_tpye = l_type
        self.ulosses = []
        self.accuracies = []
        
        unlabeled_losses = OrderedDict()
        for t in range(T-1):
            l_name = "unlabeled-loss-{:03d}".format(t)
            unlabeled_losses[l_name] = UnlabeledLoss(l_type)
        
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

