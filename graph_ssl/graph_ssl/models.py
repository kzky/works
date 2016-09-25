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

class MLP(Chain):
    """MLP
    """
    def __init__(self,
                 dims=[784, 1000, 500, 250, 250, 250, 10],
                 act=F.relu,
                 test=False):

        # Create and set links
        fc_layers = OrderedDict()
        bn_layers = OrderedDict()
        layers = {}
        for i, d in enumerate(zip(dims[:-1], dims[1:])):
            fc_name = "fc{}".format(i)
            bn_name = "bn{}".format(i)

            fc_layers[fc_name] = L.Linear(d[0], d[1])
            bn_layers[bn_name] = L.BatchNormalization(d[1])
            layers[fc_name] = fc_layers[fc_name]
            layers[bn_name] = bn_layers[bn_name]
            
        super(MLP, self).__init__(**layers)

        # Set attributes
        self.fc_layers = fc_layers
        self.bn_layers = bn_layers
        self.act = act
        self.test = False
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
        
        h = x
        for fc, bn in zip(self.fc_layers.values(), self.bn_layers.values()):
            z = fc(h)
            z_bn = bn(z, self.test)
            h = self.act(z_bn)

            #TODO: Add non-BN output
            mid_outputs.append(z)
        
        return h

class CrossEntropy(Chain):
    def __init__(self, predictor):
        super(CrossEntropy, self).__init__(predictor=predictor)
        self.loss = None
        self.accuracy = None
        
    def __call__(self, x_l, y_l):
        y = self.predictor(x_l)
        self.accuracy = F.accuracy(y, y_l)
        self.loss = F.softmax_cross_entropy(y, y_l)
        
        return self.loss

class RBF(Link):
    def __init__(self, dim):
        super(RBF, self).__init__(
            gamma=(1, dim)
        )
        self.gamma.data[:] = np.random.randn(1, dim)
        
    def __call__(self, x, y):
        g = self.gamma ** 2
        z = F.expand_dims((x - y) ** 2, axis=0)
        o = F.exp(- F.linear(z, g))
        return o
        
class GraphLoss(Chain):
    """Graph Loss

    Parameters
    -----------------
    ffnn_u_0: MLP (now)
    ffnn_u_1: MLP (now)
    """
    def __init__(self, ffnn_u_0, ffnn_u_1, dims, batch_size):
        # Create and set chain
        layers = {}
        similarities = OrderedDict()
        for i, d in enumerate(dims[1:]):
            sim_name = "sim{}".format(i+1)
            similarities[sim_name] = RBF(d)
            layers[sim_name] = similarities[sim_name]

        layers["ffnn_u_0"] = ffnn_u_0
        layers["ffnn_u_1"] = ffnn_u_1

        super(GraphLoss, self).__init__(**layers)

        # Set attributes
        self.layers = layers
        self.similarities = similarities
        self.dims = dims
        self.batch_size = batch_size
        self.coef = 1. / batch_size

    def __call__(self, x_u_0, x_u_1):
        ffnn_u_0 = self.layers["ffnn_u_0"]
        ffnn_u_1 = self.layers["ffnn_u_1"]
        
        f_0 = F.softmax(ffnn_u_0(x_u_0))
        f_1 = F.softmax(ffnn_u_1(x_u_1))

        mid_outputs_0 = ffnn_u_0.mid_outputs
        mid_outputs_1 = ffnn_u_1.mid_outputs
        
        #TODO: Compare fast matmul implementation if possible
        loss = 0
        L = len(self.dims[1:])
        similarities = self.similarities.values()
        batch_size = self.batch_size
        for i in range(batch_size - 1):
            f_0_i = f_0[i, :]

            for j in range(i, batch_size):
                f_1_j = f_1[j, :]

                s = 0  # similarity over layers, i.e., factors of variations
                for l in range(L):
                    o_0_i = mid_outputs_0[l][i, :]
                    o_1_j = mid_outputs_1[l][j, :]
                    s += similarities[l](o_0_i, o_1_j)

                # one term between i-th and j-th sample. align shape to () not (1, 1)
                loss += F.reshape(s, ()) * F.sum((f_0_i - f_1_j) ** 2)

        loss /= batch_size
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
        
    def __call__(self, x_l, y_l, x_u_0, x_u_1):
        loss = self.lambdas[0] * self.sloss(x_l, y_l) \
               + self.lambdas[1] * self.gloss(x_u_0, x_u_1)

        return loss

class GraphSSLMLPModel(Chain):
    """Graph-based Semi-Supervised Learning Model

    Class instantiates the all chains necessary for foward pass upto the objective,
    and the objective is passed to the super.__init__ method as a chain.
    """

    def __init__(self, dims, batch_size, lambdas=np.array([1., 1.])):
        # Create chains
        mlp_l = MLP(dims)
        mlp_u_0 = mlp_l.copy()
        mlp_u_1 = mlp_l.copy()
        sloss = CrossEntropy(mlp_l)
        gloss = GraphLoss(mlp_u_0, mlp_u_1, dims, batch_size)
        ssl_graph_loss = SSLGraphLoss(sloss, gloss, lambdas)

        # Set as attrirbutes for shortcut access
        self.mlp_l = mlp_l
        self.mlp_u_0 = mlp_u_0
        self.mlp_u_1 = mlp_u_1
        self.sloss = sloss
        self.gloss = gloss

        # Set chain
        super(GraphSSLMLPModel, self).__init__(ssl_graph_loss=ssl_graph_loss)

    def __call__(self, x_l, y_l, x_u_0, x_u_1):
        return self.ssl_graph_loss(x_l, y_l, x_u_0, x_u_1)
    
