from simple_ssl.models import MLPEncDecModel, NegativeEntropyLoss
from chainer import optimizers, Variable
import chainer.functions as F
from utils import grad_norm_hook
from sklearn.metrics import confusion_matrix
from chainer import cuda
import numpy as np
import cv2

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
        self.graph_loss = self.model.graph_loss
        self.kl_recon_loss = self.model.kl_recon_loss
        self.cc_loss = self.model.cc_loss

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate)
        self.optimizer.setup(self.model)
        self.optimizer.use_cleargrads()
        self.optimizer.add_hook(grad_norm_hook, "grad_norm_hook")
        
    def train(self, x_l, y_l, x_u):
        loss = self.forward(x_l, y_l, x_u)
        self.backward(loss)
        #self.optimizer.call_hooks()
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

        # Reconstruction for (x_l, _)
        y = self.mlp_enc(x_l, test)
        x_l_recon = self.mlp_dec(y, test)
        recon_loss_l = self.recon_loss(x_l_recon, x_l,  # Use self, x_l
                                       self.mlp_enc.hiddens, 
                                       self.mlp_dec.hiddens, 
                                       self.scale_rc)

        # Negative Entropy for y_l
        if self.lds:
            #TODO: add mlp_dec.hiddens?
            neg_ent_l = self.neg_ent_loss(y, self.mlp_enc.hiddens, scale=self.scale_lds) 
        else:
            neg_ent_l = self.neg_ent_loss(y, scale=self.scale_lds)
        
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

        return supervised_loss, recon_loss_u, neg_ent_u

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
        y_argmax = F.argmax(y, axis=1)
        acc = F.accuracy(y, y_l)
        y_l_cpu = cuda.to_cpu(y_l.data)
        y_argmax_cpu = cuda.to_cpu(y_argmax.data)

        # Confuction Matrix
        cm = confusion_matrix(y_l_cpu, y_argmax_cpu)
        print(cm)

        # Wrong samples
        idx = np.where(y_l_cpu != y_argmax_cpu)[0]
        #print(idx.tolist())

        # Generate and Save
        x_rec = self.mlp_dec(y, test=True)
        self.save_generate_images(x_rec, idx)

        loss = self.forward_for_losses(x_l, y_l, None, test=True)  # only measure x_l
        supervised_loss = loss
        return acc, supervised_loss

    def save_generate_images(self, x_rec, idx=None):
        if idx is not None:
            x_rec = x_rec.data[idx, :]
        else:
            x_rec = x_rec.data
            
        for i, img in enumerate(x_rec):
            fpath = "./test_gen/{:05d}.png".format(i)
            cv2.imwrite(fpath, img.reshape(28, 28) * 255.)

class Experiment1000(Experiment):
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
        self.graph_loss = self.model.graph_loss
        self.kl_recon_loss = self.model.kl_recon_loss
        self.cc_loss = self.model.cc_loss

        # Optimizer
        self.optimizer_ce = optimizers.Adam(learning_rate)
        self.optimizer_ce.setup(self.model)
        self.optimizer_ce.use_cleargrads()

        self.optimizer_rc = optimizers.Adam(learning_rate)
        self.optimizer_rc.setup(self.model)
        self.optimizer_rc.use_cleargrads()

        self.optimizer_ne = optimizers.Adam(learning_rate)
        self.optimizer_ne.setup(self.model)
        self.optimizer_ne.use_cleargrads()
                
    def train(self, x_l, y_l, x_u, test=False):
        # Supervision for (x_l, y_l)
        y = self.mlp_enc(x_l, test)
        supervised_loss = self.supervised_loss(y, y_l)
        self.model.cleargrads()
        supervised_loss.backward()
        self.optimizer_ce.update()

        # Reconstruction for (x_u, _)
        y = self.mlp_enc(x_u, test)
        x_u_recon = self.mlp_dec(y, test)
        recon_loss_u = self.recon_loss(x_u_recon, x_u,  # Use self, x_u
                                       self.mlp_enc.hiddens, 
                                       self.mlp_dec.hiddens, 
                                       self.scale_rc)
        self.model.cleargrads()
        recon_loss_u.backward()
        self.optimizer_rc.update()

        # Negative Entropy for y_u
        if self.lds:
            neg_ent_u = self.neg_ent_loss(y, self.mlp_enc.hiddens, scale=self.scale_lds) 
        else:
            neg_ent_u = self.neg_ent_loss(y, scale=self.scale_lds)
        self.model.cleargrads()
        neg_ent_u.backward()
        self.optimizer_ne.update()

# Alias
Experiment000 = Experiment
