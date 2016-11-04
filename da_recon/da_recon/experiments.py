from da_recon.models import MLPEncDecModel
from chainer import optimizers, Variable
import chainer.functions as F

class Experiment000(object):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                     device=None,
                     learning_rate=1. * 1e-2,
                     lambdas = [1., 1., 1.],
                     dims,
                     act=F.Relu,
                     bn=True,
                     noise=False,
                     lateral=False,
                     test=False,):

        # Settting
        self.devide = device
        self.lambdas = lambdas
        self.T = len(lambdas)
        
        # Model
        self.model = MLPEncDecModel(
            dims=dims, act=act,
            bn=bn, noise=noise, lateral=lateral, test=test)
        self.mlp_enc = self.model.mlp_enc
        self.mlp_dec = self.model.mlp_dec
        self.supervised_loss = self.model.supervised_loss
        self.recon_loss = self.model.recon_loss

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(model)
        self.oprimizer.use_cleargrads()
        
    def train(self, x_l, y_l, x_u):
        # Forward
        supervised_losses = []
        recon_loss_ls = []
        recon_loss_us = []

        x_l_recon = x_l
        x_u_recon = x_u
        for t in range(T):
            # Supervision for (x_l, y_l)
            y = self.mlp_enc(x_l_recon)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y)
            recon_loss_l = self.recon_loss(x_l_recon, x_l,  # Use self, x_l
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_loss_ls.append(recon_loss_l)

            # Reconstruction for (x_u, _)
            y = self.mlp_enc(x_u_recon)
            x_u_recon = self.mlp_dec(y)
            recon_loss_u = self.recon_loss(x_u_recon, x_u,  # Use self, x_u
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)        
            recon_loss_us.append(recon_loss_u)

        # Loss
        supervised_loss = reduce(lambda x, y: x + y)
        recon_loss_l = 0
        recon_loss_u = 0
        for lambda_, l0, l1 in zip(self.lambdas, recon_loss_ls, recon_loss_us):
            recon_loss_l += lambda_ * l0
            recon_loss_u += lambda_ * l1

        loss = supervised_loss + recon_loss_l + recon_loss_u

        # Backward
        self.model.cleargrads()
        loss.backward()

        # Update
        optimizer.update()

    def test(self, x_l, y_l):
        y = self.mlp_enc(x_l)
        return F.accuracy(y, y_l)
        
