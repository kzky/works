from da_recon4.models import Encoder, Decoder, Generator, Discriminator
from chainer import optimizers, Variable
import chainer.functions as F
import numpy as np
from da_recon4.utils import to_onehot

def test_encoder():
    bs = 8
    dim = 784

    encoder = Encoder()
    x = np.random.rand(bs, dim).astype(np.float32)
    h = encoder(x)

def test_decoder():
    bs = 8
    dim = 784
    n_cls = 10

    # Encoder
    encoder = Encoder()
    x = np.random.rand(bs, dim).astype(np.float32)
    h = encoder(x)

    # Decoder
    decoder = Decoder()
    y_ = np.random.choice(n_cls, bs)
    y = to_onehot(y_, n_cls)
    x_rec = decoder(h, encoder.hiddens, y)
    
def test_generator():
    bs = 8
    dim = 784
    n_cls = 10
    fix = True
     
    # Encoder
    encoder = Encoder()
    x = np.random.rand(bs, dim).astype(np.float32)
    h = encoder(x)
     
    # Decoder
    decoder = Decoder()
    y_ = np.random.choice(n_cls, bs)
    y = to_onehot(y_, n_cls)
    x_rec = decoder(h, encoder.hiddens, y)
     
    # Generator
    generator = Generator(decoder, fix=fix)
    generator(bs, y)

    
def test_discriminator():
    bs = 8
    dim = 784
    n_cls = 10
    fix = True

    # Encoder
    encoder = Encoder()
    x = np.random.rand(bs, dim).astype(np.float32)
    h = encoder(x)
     
    # Decoder
    decoder = Decoder()
    y_ = np.random.choice(n_cls, bs)
    y = to_onehot(y_, n_cls)
    x_rec = decoder(h, encoder.hiddens, y)
     
    # Generator
    generator = Generator(decoder, fix=fix)
    x_gen = generator(bs, y)

    # Discriminator
    discriminator = Discriminator()
    discriminator(x_gen)
    
