"""Models Tests
"""
import numpy as np
from chainer import optimizers
from models import MLP, CrossEntropy, RBF, GraphLoss, SSLGraphLoss, GraphSSLMLPModel

def test_loss():
    batch_size = 32
    inp_dim = 784
    out_dim = 10
    dims = [inp_dim, 1000, 500, 250, 250, 250, out_dim]

    x_l = np.random.randn(batch_size, inp_dim).astype(np.float32)
    y_l = np.random.uniform(0, out_dim, batch_size).astype(np.int32)
    x_u_0 = np.random.randn(batch_size, inp_dim).astype(np.float32)
    x_u_1 = np.random.randn(batch_size, inp_dim).astype(np.float32)

    model = GraphSSLMLPModel(dims, batch_size)
    fc0_W_data = model.mlp_l["fc0"].W.data.copy()  # once copy
        
    # Optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # Train one-step
    model.zerograds()
    loss = model.foward(x_l, y_l, x_u_0, x_u_1)
    loss.backward()
    optimizer.update()

    # Check grad computation
    assert np.all(model.mlp_l["fc0"].W.grad != 0.)

    # Check data update
    assert np.any(fc0_W_data != model.mlp_l["fc0"].W.data)

    # Check shared params
    assert np.allclose(model.mlp_l["fc0"].W.data, model.mlp_u_0["fc0"].W.data)
    
    
