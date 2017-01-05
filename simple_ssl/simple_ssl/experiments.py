from simple_ssl.models import MLPEncDecModel, NegativeEntropyLoss
from chainer import optimizers, Variable
import chainer.functions as F

class Experiment(object):
    """Experiment takes responsibility for a batch not for train-loop.
    """

    def __init__(self,
                 device=None,
                 learning_rate=1. * 1e-2,
                 dims=[784, 250, 100, 10],
                 act=F.relu,
                 noise=False,
                 rc=False,
                 lds=False,
                 scale_rc=False,
                 scale_lds=False,
                 ):

        # Settting
        self.device = device
        self.rc = rc
        self.lds = lds
        self.scale_rc = scale_rc
        self.scale_lds = scale_lds
        
        # Model
        self.model = MLPEncDecModel(
            dims=dims, act=act,
            noise=noise, rc=rc,
            device=device)
        self.model.to_gpu(self.device) if self.device else None
        self.mlp_enc = self.model.mlp_enc
        self.mlp_dec = self.model.mlp_dec
        self.supervised_loss = self.model.supervised_loss
        self.recon_loss = self.model.recon_loss
        self.neg_ent_loss = self.model.neg_ent_loss

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

        # Supervision for (x_l, y_l)
        y = self.mlp_enc(x_l, test)
        supervised_loss = self.supervised_loss(y, y_l)
        
        if x_u is None:
            return supervised_loss
            
        # Reconstruction for (x_u, _)
        y = self.mlp_enc(x_u, test)
        x_u_recon = self.mlp_dec(y, test)
        recon_loss_u = self.recon_loss(x_u_recon, x_u,  # Use self, x_u
                                       self.mlp_enc.hiddens, 
                                       self.mlp_dec.hiddens, 
                                       self.scale_rc)

        # Negative Entropy for y_u
        if self.lds:
            #TODO: add mlp_dec.hiddens?
            neg_ent_u = self.neg_ent_loss(y, self.mlp_enc.hiddens, scale=self.scale_lds) 
        else:
            neg_ent_u = self.neg_ent_loss(y, scale=self.scale_lds)

        return supervised_loss, recon_loss_l, recon_loss_u, neg_ent_l, neg_ent_u

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
        losses = self.forward_for_losses(x_l, y_l, None, test=True)  # only measure x_l
        supervised_loss = losses[0]
        recon_loss = losses[1]
        return acc, supervised_loss, recon_loss
        
# Alias
Experiment000 = Experiment
