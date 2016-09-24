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

        # Create and set layers
        fc_layers = OrderedDict()
        bn_layers = OrderedDict()
        layers = {}
        for i, dim in enumerate(zip(dims[:-1], dims[1:])):
            fc_name = "fc{}".format(i)
            bn_name = "bn{}".format(i)

            fc_layers[fc_name] = L.Linear(dim[0], dim[1])
            bn_layers[bn_name] = L.BatchNormalization(dim[1])
            layers[fc_name] = fc_layers[fc_name]
            layers[bn_name] = fc_layers[bn_name]
            
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
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            z = l(h)
            z_bn = bn(z, self.test)
            h = self.act(z_bn)
            
            mid_outputs.append(z)
            
        return h

class CrossEntropy(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x_l, y_l):
        y = self.predictor(x_l)
        accuracy = F.accuracy(y, y_l)
        loss = F.softmax_cross_entropy(y, y_l)
        report({"supervised loss": loss, "accuracy": accuracy}, self)

        return loss

class RBF(Link):
    def __init__(self, dim):
        super(RBF, self).__init__(
            gamma=(dim)
        )
        self.gamma = np.random.randn(dim)
        
    def __call__(self, x, y):
        g = F.expand_dims(self.gamma ** 2, 0)
        z = (x - y) ** 2
        o = F.exp(- F.linear(z, g))
        return o
        
class GraphLoss(Chain):
    """Graph Loss

    Parameters
    -----------------
    ffnn_u0: MLP (now)
    ffnn_u1: MLP (now)
    """
    def __init__(self, ffnn_u0, ffnn_u1, dims, batch_size):
        # Create and set layers
        layers = {}
        gammas = {}
        for i, d in enumerate(dims[1:]):
            g_name = "g{}".format(i+1)
            gammas[g_name] = RBF(d)
            layers[f_namme] = gammas[g_name]

        layers["ffnn_u0"] = ffnn_u0
        layers["fnnn_u1"] = ffnn_u1

        super(GraphLoss, self).__init__(**layers)

        # Set attributes
        self.gammas = gammas
        self.dims = dims
        self.batch_size = batch_size
        
        
    def __call__(self, x_u0, x_u1):
        f0 = F.softmax(self.ffnn_u0(x_u0))
        f1 = F.softmax(self.ffnn_u1(x_u1))

        mid_outputs_0 = self.ffnn_u0.mid_outputs_0
        mid_outputs_1 = self.ffnn_u0.mid_outputs_1
        
        #TODO: Compare fast matmul implementation if possible
        loss = 0
        L = len(self.dims[1:])
        gammas = self.gammas
        for i in range(batch_size - 1):
            o0_i = mid_output_0[i, :]
            f0_i = f0[i, :]
            for j in range(i+1, batch_size):
                o1_j = mid_outputs_1[j]
                f1_j = f1[j, :]
                
                s = 0  # similarity over layers, i.e., factors of variations
                for l in range(L):
                    s += gammas[l](o0_i, o1_j)

                loss += s * F.sum((f0_i - f1_j) ** 2)
                    
        return loss

class Loss(Chain):
    """Loss function, objective

    Parameters
    -----------------
    sloss: CrossEntropy
    gloss: GraphLoss
    lambdas: list
         Coefficients between supervised loss and graph loss
    """
    def __init__(self, sloss, gloss, lambdas=[1., 1.]):
        super(Loss, self).__init__(sloss=sloss, gloss=gloss)

        #self.lambdas = [Varialbe(l).to_gpu() for l in lambdas]
        self.lambdas = lambdas
        
    def __call__(self, x_l, y_l, x_u0, x_u1):
        loss = self.lambdas[0] * self.sloss(x_l, y_l) \
               + self.lambdas[1] * self.gloss(x_u0, x_u1)

        return loss
