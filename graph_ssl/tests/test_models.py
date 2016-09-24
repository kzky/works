"""Models Tests
"""
import numpy as np
from models import MLP, CrossEntropy, RBF, GraphLoss, SSLGraphLoss

def test_loss():
    batch_size = 32
    inp_dim = 784
    out_dim = 10
    dims = [inp_dim, 1000, 500, 250, 250, 250, out_dim]

    x_l = np.random.rand(batch_size, inp_dim).astype(np.float32)
    y_l = np.random.uniform(0, out_dim, batch_size).astype(np.int32)
    x_u_0 = np.random.rand(batch_size, inp_dim).astype(np.float32)
    x_u_1 = np.random.rand(batch_size, inp_dim).astype(np.float32)

    mlp_l = MLP(dims)
    mlp_u_0 = mlp_l.copy()  # weight tying
    mlp_u_1 = mlp_l.copy()  # weight tying
    sloss = CrossEntropy(mlp_l)
    gloss = GraphLoss(mlp_u_0, mlp_u_1, dims, batch_size)
    model = SSLGraphLoss(sloss, gloss)

    # Forward
    loss = model(x_l, y_l, x_u_0, x_u_1)

    # Backward
    loss.zerograd()
    loss.backward()

    print(loss.data)
    

    
