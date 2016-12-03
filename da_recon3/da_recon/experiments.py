from da_recon.models import MLPEncDecModel
from chainer import optimizers, Variable
import chainer.functions as F

class Experiment(object):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 lambdas = [1., 1., 1.],
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 noise=False,
                 rc=False,
                 lateral=False):

        # Settting
        self.device = device
        self.lambdas = lambdas
        self.T = len(lambdas)
        
        # Model
        self.model = MLPEncDecModel(
            dims=dims, act=act,
            noise=noise, rc=rc, lateral=lateral,
            device=device)
        self.model.to_gpu(self.device) if self.device else None
        self.mlp_enc = self.model.mlp_enc
        self.mlp_dec = self.model.mlp_dec
        self.supervised_loss = self.model.supervised_loss
        self.recon_loss = self.model.recon_loss

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
        
    def train(self, x_l, y_l, x_u):
        loss = self.forward(x_l, y_l, x_u)
        self.backward(loss)
        self.update()

    def forward_for_losses(self, x_l, y_l, x_u, test=False):
        """
        Returns
        -----------
        tuple:
            tuple of Variables for separate loss
        """

        # Forward
        supervised_losses = []
        recon_l_losses = []
        recon_u_losses = []

        x_l_recon = x_l
        x_u_recon = x_u
        for t in range(self.T):
            # Supervision for (x_l, y_l)
            y = self.mlp_enc(x_l_recon, test)
            
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y, test)
            recon_loss_l = self.recon_loss(x_l_recon, x_l,  # Use self, x_l
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)
                
            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                continue

            y = self.mlp_enc(x_u_recon, test)
            x_u_recon = self.mlp_dec(y, test)
            recon_loss_u = self.recon_loss(x_u_recon, x_u,  # Use self, x_u
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)
            recon_u_losses.append(recon_loss_u)

        # Loss
        supervised_loss = reduce(lambda x, y: x + y, supervised_losses)

        recon_loss_l = 0
        recon_loss_u = 0
        for lambda_, l0, l1 in zip(self.lambdas,  # Use coefficients for ulosses
                                       recon_l_losses,
                                       recon_u_losses):
            recon_loss_l += lambda_ * l0
            recon_loss_u += lambda_ * l1

        return supervised_loss, recon_loss_l, recon_loss_u

    def forward(self, x_l, y_l, x_u, test=False):
        losses = self.forward_for_losses(x_l, y_l, x_u)
        return reduce(lambda x, y: x + y, losses)

    def backward(self, loss):
        self.model.cleargrads()
        loss.backward()

    def update(self, ):
        self.optimizer.update()

    def test(self, x_l, y_l):
        y = self.mlp_enc(x_l, test=True)
        acc = F.accuracy(y, y_l)
        losses = self.forward_for_losses(x_l, y_l, None, test=True)
        supervised_loss = losses[0]
        recon_loss = losses[1]
        return acc, supervised_loss, recon_loss
        
class Experiment070_3(Experiment):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 lambdas = [1., 1., 1.],
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 noise=False,
                 rc=False,
                 lateral=False):
        
        super(Experiment070_3, self).__init__(
            device=device,
            learning_rate=learning_rate,
            lambdas=lambdas,
            dims=dims,
            act=act,
            noise=noise,
            rc=rc,
            lateral=lateral)

        self.kl_loss = self.model.kl_loss

    def forward_for_losses(self, x_l, y_l, x_u, test=False):
        """
        Returns
        -----------
        tuple:
            tuple of Variables for separate loss
        """

        # Forward
        supervised_losses = []
        kl_l_losses = []
        kl_u_losses = []
        recon_l_losses = []
        recon_u_losses = []

        x_l_recon_t0 = x_l
        x_u_recon_t0 = x_u
        
        for t in range(self.T):
            # Supervision for (x_l, y_l)
            y = self.mlp_enc(x_l_recon_t0, test)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            if t > 1:  # KL Divergence Loss
                kl_l_loss = self.kl_loss(y, y_p_l)
                kl_l_losses.append(kl_l_loss)
            y_p_l = y
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y, test)
            recon_loss_l = self.recon_loss(x_l_recon, x_l_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)
            x_l_recon_t0 = x_l_recon

            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                kl_l_losses.append(0)
                kl_u_losses.append(0)
                continue

            y = self.mlp_enc(x_u_recon_t0, test)
            x_u_recon = self.mlp_dec(y, test)
            recon_loss_u = self.recon_loss(x_u_recon,  x_u_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)        
            recon_u_losses.append(recon_loss_u)
            x_u_recon_t0 = x_u_recon

            if t > 1:  # KL Divergence Loss
                kl_u_loss = self.kl_loss(y, y_p_u)
                kl_u_losses.append(kl_l_loss)
            y_p_u = y

        # Loss
        supervised_loss = reduce(lambda x, y: x + y, supervised_losses)

        recon_loss_l = 0
        recon_loss_u = 0
        for lambda_, l0, l1 in zip(self.lambdas,  # Use coefficients for ulosses
                                       recon_l_losses,
                                       recon_u_losses):
            recon_loss_l += lambda_ * l0
            recon_loss_u += lambda_ * l1

        return supervised_loss, recon_loss_l, recon_loss_u, kl_loss

class Experiment000(Experiment):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 lambdas = [1., 1., 1.],
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 noise=False,
                 rc=False,
                 lateral=False):

        super(Experiment000, self).__init__(
            device=device,
            learning_rate=learning_rate,
            lambdas=lambdas,
            dims=dims,
            act=act,
            noise=noise,
            rc=rc,
            lateral=lateral)

        self.pseudo_supervised_loss = self.model.pseudo_supervised_loss

    def forward_for_losses(self, x_l, y_l, x_u, test=False):
        """
        Returns
        -----------
        tuple:
            tuple of Variables for separate loss
        """

        # Forward
        supervised_losses = []
        pseudo_supervised_l_losses = []
        pseudo_supervised_u_losses = []
        recon_l_losses = []
        recon_u_losses = []

        x_l_recon_t0 = x_l
        x_u_recon_t0 = x_u
        
        for t in range(self.T):
            # Supervision for (x_l, y_l)
            y = self.mlp_enc(x_l_recon_t0, test)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            if t > 1:  # Pseudo Supervised Loss
                pseudo_supervised_l_loss = self.pseudo_supervised_loss(y, y_p_l)
                pseudo_supervised_l_losses.append(pseudo_supervised_l_loss)
            y_p_l = y
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y, test)
            recon_loss_l = self.recon_loss(x_l_recon, x_l_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)
            x_l_recon_t0 = x_l_recon

            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                pseudo_supervised_l_losses.append(0)
                pseudo_supervised_u_losses.append(0)
                continue

            y = self.mlp_enc(x_u_recon_t0, test)
            x_u_recon = self.mlp_dec(y, test)
            recon_loss_u = self.recon_loss(x_u_recon,  x_u_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)        
            recon_u_losses.append(recon_loss_u)
            x_u_recon_t0 = x_u_recon

            if t > 1:  # Pseudo Supervised Loss
                pseudo_supervised_u_loss = self.pseudo_supervised_loss(y, y_p_u)
                pseudo_supervised_u_losses.append(pseudo_supervised_l_loss)
            y_p_u = y

        # Loss
        supervised_loss = reduce(lambda x, y: x + y, supervised_losses)

        recon_loss_l = 0
        recon_loss_u = 0
        for lambda_, l0, l1 in zip(self.lambdas,  # Use coefficients for ulosses
                                       recon_l_losses,
                                       recon_u_losses):
            recon_loss_l += lambda_ * l0
            recon_loss_u += lambda_ * l1

        # Pseudo Supervised Loss
        pseudo_supervised_loss = 0
        for lambda_, l0, l1 in zip(self.lambdas, pseudo_supervised_l_losses,
                                       pseudo_supervised_u_losses):
            pseudo_supervised_loss += lambda_ * (l0 + l1)

        return supervised_loss, recon_loss_l, recon_loss_u, pseudo_supervised_loss


# Alias
Experiment001 = Experiment000
Experiment005 = Experiment000

