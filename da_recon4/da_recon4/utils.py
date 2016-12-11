import numpy as np
import os
from chainer import cuda

def to_device(x, device=None):
    if device:
        return cuda.to_gpu(x, device)
    else:
        return cuda.to_cpu(x)

def to_onehot(y_, n_cls):
    y = np.zeros((len(y_), n_cls)).astype(np.float32)
    y[np.arange(len(y_)), y_] = 1.
    return y
    
    
