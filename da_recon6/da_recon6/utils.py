import numpy as np
import os
from chainer import cuda, Variable
import chainer.functions as F
import cv2

def to_device(x, device=None):
    if device:
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

def grad_unbias_hook(optimizer):
    for p in optimizer.target.params():
        grad_data = p.grad
        bs = grad_data.shape[0]
        shape = grad_data.shape
        
        grad = Variable(grad_data)
        mean_grad = F.broadcast_to(F.sum(grad) / bs, shape)
        grad_unbias = grad - mean_grad

        p.grad = grad_unbias.data
        
def save_generate_images(x_rec, idx=None):
        if idx is not None:
            x_rec = x_rec.data[idx, :]
        else:
            x_rec = x_rec.data

        if os.path.exists("./test_gen"):
            shutil.rmtree("./test_gen")
            os.mkdir("./test_gen")
        else:
            os.mkdir("./test_gen")
            
        for i, img in enumerate(x_rec):
            fpath = "./test_gen/{:05d}.png".format(i)
            cv2.imwrite(fpath, img.reshape(28, 28) * 255.)
