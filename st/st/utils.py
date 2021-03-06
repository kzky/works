import numpy as np
import os
from chainer import cuda, Variable
import chainer.functions as F
import cv2
import shutil
import csv

def to_device(x, device=None):
    if device is not None:
        return cuda.to_gpu(x, device)
    else:
        return cuda.to_cpu(x)

def add_normal_noise(h, sigma=0.03):
    if np.random.randint(0, 2):
        n = np.random.normal(0, sigma, h.data.shape).astype(np.float32)
        device = cuda.get_device(h)
        if device.id ==  -1:
            n_ = Variable(n)
        else:
            n_ = Variable(cuda.to_gpu(n, device.id))
        h = h + n_
    return h
