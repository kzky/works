from simple_ssl2.models import MLPAE
from simple_ssl2.losses import ReconstructionLoss, NegativeEntropyLoss
from chainer import optimizers, Variable
from chainer import serializers
import chainer.functions as F
from utils import grad_norm_hook, grad_unbias_hook
from utils import save_incorrect_info
from sklearn.metrics import confusion_matrix
from chainer import cuda
import numpy as np
import cv2

class Experiment(object):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,                 
                 act=F.relu,
                 learning_rate=1. * 1e-2,
                 ):

        # Settting
        self.device = device
        self.act = act
        self.learning_rate = learning_rate

        # Model
        self.mlp_ae = MLPAE(device, act)

        # Loss
        self.rc_loss = ReconstructionLoss()
        self.ne_loss = NegativeEntropyLoss()

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.use_cleargrads()
        self.optimizer.setup(mlp_ae)

    def train(self, x_l, y_l, x_u):
        loss = self.forward(x_l, y_l, x_u)
        self.mlp_ae.cleargrads()        
        loss.backward()
        self.optimizer.update()

    def forward(self,  x_l, y_l, x_u):
        # Labeled data
        ## Cross Entropy
        y, z = self.mlp_ae.mlp_encoder(x_l)
        loss_ce_l = F.softmax_cross_entropy(y, y_l)

        ## Negative Entropy
        loss_ne_l = self.ne_loss(y)

        ## Reconstruction
        x_recon = self.mlp_ae.mlp_decocer(z)
        loss_rc_l = self.rc_loss(x_recon, x)

        # Unlabeled data
        ## Cross Entropy
        y, z = self.mlp_ae.mlp_encoder(x_u)

        ## Negative Entropy
        loss_ne_u = self.ne_loss(y)

        ## Reconstruction
        x_recon = self.mlp_ae.mlp_decocer(z)
        loss_rc_u = self.rc_loss(x_recon, x)

        loss = loss_ce_l + loss_ne_l + loss_rc_l + loss_ne_u + loss_rc_u
        return loss
        
    def test(self, x_l, y_l):
        y = F.softmax(self.mlp_ae.mlp_encoder(x_l, test=True))
        y_argmax = F.argmax(y, axis=1)
        acc = F.accuracy(y, y_l)
        y_l_cpu = cuda.to_cpu(y_l.data)
        y_argmax_cpu = cuda.to_cpu(y_argmax.data)

        # Confuction Matrix
        cm = confusion_matrix(y_l_cpu, y_argmax_cpu)
        print(cm)

        # Wrong samples
        idx = np.where(y_l_cpu != y_argmax_cpu)[0]

        # Generate and Save
        x_rec = self.mlp_ae.mlp_decoder(y, self.mlp_encoder.hiddens, test=True)
        save_incorrect_info(x_rec.data[idx, ], x_l.data[idx, ],
                            y.data[idx, ], y_l.data[idx, ])

        # Save model
        serializers.save_hdf5("./model/mlp_encdec.h5py", self.mlp_ae)

        return acc

