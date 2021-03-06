"""Models
"""
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from utils import to_device

class MLP(Chain):
    """MLP

    Parameters
    -----------------
    dims: list of int
        Each element corresponds to the units. 
    act: activation
        E.g,.  F.relu or F.tanh
    decay: float
        Decay parameter of Batch Normlization
    test: bool
        If False, the running mean and variance are computed in Batch Normalization,
        and batch mean and variacne are used in batch normliazatoin; otherwise 
        in the inference time, the comupted running mean and variance are used in
        Batch Normalization.
    """
    def __init__(self,
                 dims=[784, 1000, 500, 250, 250, 250, 10],
                 act=F.tanh,
                 decay=0.9,
                 device=None):

        # Create and set links
        fc_layers = OrderedDict()
        bn_layers = OrderedDict()
        layers = {}
        for i, d in enumerate(zip(dims[:-1], dims[1:])):
            fc_name = "fc{}".format(i)
            bn_name = "bn{}".format(i)
            dp_name = "dp{}".format(i)
            fc_layers[fc_name] = L.Linear(d[0], d[1])
            bn_layers[bn_name] = L.BatchNormalization(d[1], decay)
            layers[fc_name] = fc_layers[fc_name]
            layers[bn_name] = bn_layers[bn_name]
            
        super(MLP, self).__init__(**layers)

        # Set attributes
        self.fc_layers = fc_layers
        self.bn_layers = bn_layers
        self.act = act
        self.decay = decay
        self.test = False
        self.device = device
        self.mid_outputs = []

    def __call__(self, x):
        """
        Parameters
        -----------------
        x: Variable
            Shape is 784 in case of MNIST
        """
        # Reset mid outputs
        mid_outputs = self.mid_outputs = []
        
        h  = x
        for fc, bn in zip(self.fc_layers.values(), self.bn_layers.values()):
            z = fc(h)
            z_bn = bn(z, self.test)
            h = self.act(z_bn)
            
            #TODO: Add non-BN output
            mid_outputs.append(h)

        return h

class CrossEntropy(Chain):
    """CrossEntropy Loss

    Parameters
    -----------------
    classifier: Chain
        Chain of MLP or CNN as a predictor network.
    """
    def __init__(self, predictor):
        super(CrossEntropy, self).__init__(predictor=predictor)
        self.loss = None
        self.accuracy = None
        self.pred = None
    def __call__(self, x_l, y_l):
        """
        Parameters
        -----------------
        x_l: Variable
            Feature of labeled samples.
        y_l: Variable
            Label.
        """
        y = self.predictor(x_l)
        self.pred = y
        self.accuracy = F.accuracy(y, y_l)
        self.loss = F.softmax_cross_entropy(y, y_l)

        return self.loss

class RBF0(Link):
    """RBF Kernel

    Different prameters for different dimensions.
    Efficient computation of RBF Kernel, while it consumes memory.

    Parameters
    -----------------
    dim: int
    """
    def __init__(self, dim):
        
        super(RBF0, self).__init__(
            gamma=(1, dim)
        )
        self.gamma.data[:] = np.random.normal(0, 0.1, (1, dim))

    def __call__(self, x, y):
        """
        Parameters
        -----------------
        x: Variable
            Feature of unlabeled samples.
        y: Variable
            Feature of unlabeled samples.
        """
        
        g, x, y = F.broadcast(*[self.gamma, x, y])
        x_g = x * g
        y_g = y * g

        x_g_norm = F.sum(x_g**2, axis=1)  
        y_g_norm = F.sum(y_g**2, axis=1)
        x_g_y_g = F.linear(x_g, y_g)
        
        x_g_norm, x_g_y_g, y_g_norm = \
                                      F.broadcast(
                                          *[x_g_norm,
                                            x_g_y_g,
                                            F.expand_dims(y_g_norm, 1)])
        
        return F.exp(- x_g_norm + 2 * x_g_y_g - y_g_norm)
        
class GraphLoss0(Chain):
    """Graph Loss0

    The same as GraphLoss except for using RBF0 and efficient computation,
    when computing \sum_{i, j} (f_i - f_j)^2

    Parameters
    -----------------
    classifier: MLP (now)
    dims: list of int
        Each element corresponds to the units.
    batch_size: int
    """
    def __init__(self, classifier, dims, batch_size):
        # Create and set chain
        layers = {}
        similarities = OrderedDict()
        for i, d in enumerate(dims[1:]):
            sim_name = "sim{}".format(i+1)
            similarities[sim_name] = RBF0(d)
            layers[sim_name] = similarities[sim_name]

        layers["classifier"] = classifier

        super(GraphLoss0, self).__init__(**layers)

        # Set attributes
        self.layers = layers
        self.similarities = similarities
        self.dims = dims
        self.batch_size = batch_size
        self.coef = 1. / batch_size
        self.loss = None

    def __call__(self, x_l, y_l, x_u):
        """
        Parameters
        -----------------
        x_l: Variable
            Feature of unlabeled samples.
        y_l: Variable
            Feature of unlabeled samples.
        x_u: Variable
            Feature of unlabeled samples.
        """
        classifier = self.layers["classifier"]
        f_0 = F.softmax(classifier(x_l))
        mid_outputs_0 = classifier.mid_outputs
        f_1 = F.softmax(classifier(x_u))
        mid_outputs_1 = classifier.mid_outputs
        
        L = len(self.dims[1:])
        similarities = self.similarities.values()

        # Sample similarity W^l summed over l
        W = 0
        for l in range(L):
            W += similarities[l](mid_outputs_0[l], mid_outputs_1[l])

        # Class similarity 
        f_0_norm = F.sum(f_0**2, axis=1)
        f_1_norm = F.sum(f_1**2, axis=1)
        f_0_f_1 = F.linear(f_0, f_1)
        f_0_norm, f_0_f_1, f_1_norm = \
                                      F.broadcast(
                                          *[f_0_norm,
                                            f_0_f_1,
                                            F.expand_dims(f_1_norm, 1)])
        F_ = f_0_norm - 2 * f_0_f_1 + f_1_norm

        loss = F.sum(W * F_) / (self.batch_size ** 2)
        self.loss = loss
        
        return loss

class RBF1(Link):
    """RBF Kernel

    Same prameters for different dimensions.
    Efficient computation of RBF Kernel, while it consumes memory.

    Parameters
    -----------------
    dim: int
    """
    def __init__(self, dim):
        
        super(RBF1, self).__init__(
            gamma=(1, )
        )
        self.gamma.data[:] = np.random.normal(0, 0.1, (1, ))

    def __call__(self, x, y):
        """
        Parameters
        -----------------
        x: Variable
            Feature of unlabeled samples.
        y: Variable
            Feature of unlabeled samples.
        """
        
        g, x, y = F.broadcast(*[self.gamma, x, y])
        x_g = x * g
        y_g = y * g

        x_g_norm = F.sum(x_g**2, axis=1)  
        y_g_norm = F.sum(y_g**2, axis=1)
        x_g_y_g = F.linear(x_g, y_g)
        x_g_norm, x_g_y_g, y_g_norm = \
                                      F.broadcast(
                                          *[x_g_norm,
                                            x_g_y_g,
                                            F.expand_dims(y_g_norm, 1)])
                
        return F.exp(- x_g_norm + 2 * x_g_y_g - y_g_norm)
        
class GraphLoss1(Chain):
    """Graph Loss1

    The same as GraphLoss except for using RBF1 and efficient computation,
    when computing \sum_{i, j} (f_i - f_j)^2

    Parameters
    -----------------
    classifier: MLP (now)
    dims: list of int
        Each element corresponds to the units.
    batch_size: int
    """
    def __init__(self, classifier, dims, batch_size):
        # Create and set chain
        layers = {}
        similarities = OrderedDict()
        for i, d in enumerate(dims[1:]):
            sim_name = "sim{}".format(i+1)
            similarities[sim_name] = RBF1(d)
            layers[sim_name] = similarities[sim_name]

        layers["classifier"] = classifier

        super(GraphLoss1, self).__init__(**layers)

        # Set attributes
        self.layers = layers
        self.similarities = similarities
        self.dims = dims
        self.batch_size = batch_size
        self.coef = 1. / batch_size
        self.loss = None

    def __call__(self, x_l, y_l, x_u, y_l_float32):
        """
        Parameters
        -----------------
        x_l: Variable
            Feature of unlabeled samples.
        y_l: Variable
            Label.
        x_u: Variable
            Feature of unlabeled samples.
        """

        classifier = self.layers["classifier"]
        f_0 = F.softmax(classifier(x_l))
        mid_outputs_0 = classifier.mid_outputs
        f_1 = F.softmax(classifier(x_u))
        mid_outputs_1 = classifier.mid_outputs

        L = len(self.dims[1:])
        similarities = self.similarities.values()

        # Sample similarity W^l summed over l
        W = 0
        for l in range(L):
            W += similarities[l](mid_outputs_0[l], mid_outputs_1[l])

        # Class similarity
        f_0_norm = F.sum(f_0**2, axis=1)
        f_1_norm = F.sum(y_l_float32**2, axis=1)
        f_0_f_1 = F.linear(f_0, y_l_float32)
        f_0_norm, f_0_f_1, f_1_norm = \
                                      F.broadcast(
                                          *[f_0_norm,
                                            f_0_f_1,
                                            F.expand_dims(f_1_norm, 1)])
        F_ = f_0_norm -2. * f_0_f_1 + f_1_norm
        #print(np.max(F_.data))
        #print(np.min(F_.data))
        #print(len((np.where(F_.data < 0)[0])), np.prod(F_.data.shape))
        # 
        #time.sleep(0.5)
        
        loss = F.sum(W * F_) / (self.batch_size ** 2)
        self.loss = loss

        return loss

class SSLGraphLoss(Chain):
    """Semi-Supervised Learning Graph Loss function, objective

    Parameters
    -----------------
    sloss: CrossEntropy
    gloss: GraphLoss
    lambdas: list
         Coefficients between supervised loss and graph loss
    """
    def __init__(self, sloss, gloss, lambdas=np.array([1., 1.])):
        super(SSLGraphLoss, self).__init__(sloss=sloss, gloss=gloss)

        #TODO: this should be to_gpu?
        #self.lambdas = [Variable(l).to_gpu() for l in lambdas]
        self.lambdas = lambdas
        
    def __call__(self, x_l, y_l, x_u, y_l_float32):
        """
        Parameters
        -----------------
        x_l: Variable
            Feature of labeled samples.
        y_l: Variable
            Label.
        x_u: Variable
            Feature of unlabeled samples.
        """
        loss = self.lambdas[0] * self.sloss(x_l, y_l) \
               + self.lambdas[1] * self.gloss(x_l, y_l, x_u, y_l_float32)

        return loss

class GraphSSLMLPModel(Chain):
    """Graph-based Semi-Supervised Learning Model

    Class instantiates the all chains necessary for foward pass upto the objective,
    and the objective is passed to the super.__init__ method as a chain.

    Parameters
    -----------------
    dims: list of int
    batch_size: int
    lambdas: np.ndarray
        Np.ndarray with size of two. Each is a coefficients between labeled objective
        and unlabeled objective
    
    """

    def __init__(self, dims, batch_size, act=F.relu, decay=0.9,
                 lambdas=np.array([1., 1.]), device=None):

        # Create chains
        classifier = MLP(dims, act, decay, device)
        classifier_u = classifier.copy()
        sloss = CrossEntropy(classifier)
        gloss = GraphLoss1(classifier_u, dims, batch_size)
        ssl_graph_loss = SSLGraphLoss(sloss, gloss, lambdas)

        # Set as attrirbutes for shortcut access
        self.classifier = classifier
        self.sloss = sloss
        self.gloss = gloss

        # Set chain
        super(GraphSSLMLPModel, self).__init__(ssl_graph_loss=ssl_graph_loss)

    def __call__(self, x_l, y_l, x_u, y_l_float32):
        """
        Parameters
        -----------------
        x_l: Variable
            Feature of labeled samples.
        y_l: Variable
            Label.
        x_u: Variable
            Feature of unlabeled samples.

        """
        return self.ssl_graph_loss(x_l, y_l, x_u, y_l_float32)
    
