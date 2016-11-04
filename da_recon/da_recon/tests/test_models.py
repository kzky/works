from da_recon.models import MLPEncDecModel
from chainer import optimizers, Variable
import chainer.functions as F
import numpy as np

def test_model():
    # Settings
    batch_size = 16
    dims = [784, 100, 10]
    act = F.relu
    bn = True
    noise = False
    lateral = False
    test = False

    # Model
    model = MLPEncDecModel(
        dims=dims, act=act,
        bn=bn, noise=noise, lateral=lateral, test=test)
    mlp_enc = model.mlp_enc
    mlp_dec = model.mlp_dec
    supervised_loss = model.supervised_loss
    recon_loss = model.recon_loss

    # Data
    x = np.random.rand(batch_size, dims[0]).astype(np.float32)
    y = (np.random.rand(batch_size) * 10).astype(np.int32)

    # Forward Enc/Dec
    y_pred = mlp_enc(x)
    supervised_loss = model.supervised_loss(y_pred, y)
    x_recon = model.mlp_dec(y_pred)
    recon_loss = model.recon_loss(x_recon, x, mlp_enc.hiddens, mlp_dec.hiddens)
    loss = supervised_loss + recon_loss
        
    # Backward from the root
    loss.backward()
    
    
