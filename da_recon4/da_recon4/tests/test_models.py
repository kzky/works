from da_recon4.models import MLPGenerator, MLPEncoder, MLPDecoder
from chainer import optimizers, Variable
import chainer.functions as F
import numpy as np

def test_mlp_generator():
    # Settings
    device = None
    batch_size = 16
    act = F.relu
    decay = 0.5
    act = F.relu
    sigma = 0.3
    dim = 100
    noise = False
    ncls = 10

    mlp_gen = MLPGenerator(act, sigma, device)
    y = None
    x = mlp_gen(batch_size, dim, y=y)
    y = np.random.rand(batch_size, ncls).astype(np.float32)
    x = mlp_gen(batch_size, dim, y=y)

def test_mlp_encoder():
    # Settings
    device = None
    batch_size = 16
    act = F.relu
    decay = 0.5
    act = F.relu
    sigma = 0.3
    dim = 100
    noise = False
    ncls = 10

    mlp_enc = MLPEncoder(act, sigma, device)

    x = np.random.randn(batch_size, 784).astype(np.float32)
    h = mlp_enc(x)
    

def test_mlp_decoder():
    # Settings
    device = None
    batch_size = 16
    act = F.relu
    decay = 0.5
    act = F.relu
    sigma = 0.3
    dim = 100
    noise = False
    ncls = 10

    mlp_dec = MLPDecoder(act, device)
    h = np.random.randn(batch_size, 100).astype(np.float32)
    x = mlp_dec(h)
    
