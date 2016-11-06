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
                 bn=False,
                 lateral=False,
                 test=False,):

        # Settting
        self.device = device
        self.lambdas = lambdas
        self.T = len(lambdas)
        
        # Model
        self.model = MLPEncDecModel(
            dims=dims, act=act,
            noise=noise, nb=bn, lateral=lateral, test=test, device=device)
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

    def forward_for_losses(self, x_l, y_l, x_u):
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
            y = self.mlp_enc(x_l_recon)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y)
            recon_loss_l = self.recon_loss(x_l_recon, x_l,  # Use self, x_l
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)

            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                continue
            y = self.mlp_enc(x_u_recon)
            x_u_recon = self.mlp_dec(y)
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

    def forward(self, x_l, y_l, x_u):
        losses = self.forward_for_losses(x_l, y_l, x_u)
        return reduce(lambda x, y: x + y, losses)

    def backward(self, loss):
        self.model.cleargrads()
        loss.backward()

    def update(self, ):
        self.optimizer.update()

    def set_test(self,):
        self.mlp_enc.test = True
        self.mlp_dec.test = True
        
    def unset_test(self,):
        self.mlp_enc.test = False
        self.mlp_dec.test = False
        
    def test(self, x_l, y_l):
        self.set_test()
        y = self.mlp_enc(x_l)
        acc = F.accuracy(y, y_l)
        losses = self.forward_for_losses(x_l, y_l, None)
        supervised_loss = losses[0]
        recon_loss = losses[1]
        self.unset_test()
        return acc, supervised_loss, recon_loss
        
class Experiment005(Experiment):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 lambdas = [1., 1., 1.],
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 bn=False,
                 noise=False,
                 bn=False,                 
                 lateral=False,
                 test=False,):

        super(Experiment005, self).__init__(
            device=device,
            learning_rate=learning_rate,
            lambdas=lambdas,
            dims=dims,
            act=act,
            noise=noise,
            bn=bn,
            lateral=lateral,
            test=test,)

    def forward_for_losses(self, x_l, y_l, x_u):
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
            y = self.mlp_enc(x_l_recon)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y)
            recon_loss_l = self.recon_loss(x_l_recon, x_l,  # Use self, x_l
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)
            x_l_recon = Variable(x_l_recon.data)

            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                continue
            y = self.mlp_enc(x_u_recon)
            x_u_recon = self.mlp_dec(y)
            recon_loss_u = self.recon_loss(x_u_recon, x_u,  # Use self, x_u
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)        
            recon_u_losses.append(recon_loss_u)
            x_u_recon = Variable(x_u_recon.data)
            
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

class Experiment006(Experiment):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 lambdas = [1., 1., 1.],
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 noise=False,
                 bn=False,
                 lateral=False,
                 test=False,):

        super(Experiment006, self).__init__(
            device=device,
            learning_rate=learning_rate,
            lambdas=lambdas,
            dims=dims,
            act=act,
            noise=noise,
            bn=bn,
            lateral=lateral,
            test=test,)

    def forward_for_losses(self, x_l, y_l, x_u):
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

        x_l_recon_t0 = x_l
        x_u_recon_t0 = x_u
        for t in range(self.T):
            # Supervision for (x_l, y_l)
            y = self.mlp_enc(x_l_recon_t0)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y)
            recon_loss_l = self.recon_loss(x_l_recon, x_l_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)
            x_l_recon_t0 = x_l_recon

            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                continue
            y = self.mlp_enc(x_u_recon_t0)
            x_u_recon = self.mlp_dec(y)
            recon_loss_u = self.recon_loss(x_u_recon,  x_u_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)        
            recon_u_losses.append(recon_loss_u)
            x_u_recon_t0 = x_u_recon

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

class Experiment007(Experiment):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 lambdas = [1., 1., 1.],
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 noise=False,
                 bn=False,
                 lateral=False,
                 test=False,):

        super(Experiment007, self).__init__(
            device=device,
            learning_rate=learning_rate,
            lambdas=lambdas,
            dims=dims,
            act=act,
            noise=noise,
            bn=bn,
            lateral=lateral,
            test=test,)

    def forward_for_losses(self, x_l, y_l, x_u):
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

        x_l_recon_t0 = x_l
        x_u_recon_t0 = x_u
        for t in range(self.T):
            # Supervision for (x_l, y_l)
            y = self.mlp_enc(x_l_recon_t0)
            supervised_loss = self.supervised_loss(y, y_l)
            supervised_losses.append(supervised_loss)
            
            # Reconstruction for (x_l, )
            x_l_recon = self.mlp_dec(y)
            recon_loss_l = self.recon_loss(x_l_recon, x_l_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens,
                                               self.mlp_dec.hiddens)
            recon_l_losses.append(recon_loss_l)
            x_l_recon_t0 = Variable(x_l_recon.data)

            # Reconstruction for (x_u, _)
            if x_u is None:
                recon_u_losses.append(0)
                continue
            y = self.mlp_enc(x_u_recon_t0)
            x_u_recon = self.mlp_dec(y)
            recon_loss_u = self.recon_loss(x_u_recon,  x_u_recon_t0,  # Virtual AE
                                               self.mlp_enc.hiddens, 
                                               self.mlp_dec.hiddens)        
            recon_u_losses.append(recon_loss_u)
            x_u_recon_t0 = Variable(x_u_recon.data)

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

# Alias
Experiment021 = Experiment005
Experiment022 = Experiment006
Experiment023 = Experiment007
