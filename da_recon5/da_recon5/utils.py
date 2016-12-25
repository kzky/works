import numpy as np
import os
from chainer import cuda

def to_device(x, device=None):
    if device:
        return cuda.to_gpu(x, device)
    else:
        return cuda.to_cpu(x)

def grad_norm_hook(optimizer):
    for p in optimizer.target.params():
        grad_data = p.grad
        shape = grad_data.shape
        reshape = (1, np.prod(shape), )

        grad = Variable(grad_data)
        grad_reshape = F.reshape(grad, reshape)
        grad_norm = F.normalize(grad_reshape)
        grad_norm_reshape = F.reshape(grad_norm, shape)

        p.grad = grad_norm_reshape.data
