from da_recon7.models import MLPEncDecModel
from chainer import optimizers, Variable
import chainer.functions as F
import numpy as np

def test_model():
    # Settings
    batch_size = 16
    dims = [784, 100, 10]
    act = F.relu
    noise = True
    lateral = False
    test = False

    # Model
    model = MLPEncDecModel(
        dims=dims, act=act,
        noise=noise, lateral=lateral)
    mlp_enc = model.mlp_enc
    mlp_dec = model.mlp_dec
    supervised_loss = model.supervised_loss
    recon_loss = model.recon_loss

    # Data
    x = Variable(np.random.rand(batch_size, dims[0]).astype(np.float32))
    y = Variable((np.random.rand(batch_size) * 10).astype(np.int32))

    # Forward Enc/Dec
    y_pred = mlp_enc(x, test)
    supervised_loss = model.supervised_loss(y_pred, y)
    x_recon = model.mlp_dec(y_pred, test)
    recon_loss = model.recon_loss(x_recon, x, mlp_enc.hiddens, mlp_dec.hiddens)
    neg_ent_loss = model.neg_ent_loss(y_pred)
    loss = supervised_loss + recon_loss
        
    # Backward from the root
    loss.backward()
    
    
