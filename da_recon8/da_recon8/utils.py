import numpy as np
import os
from chainer import cuda, Variable
import chainer.functions as F
import cv2
import shutil
import csv

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

def normalize_linearly(self, h):
    """Normalize h linearly in [0, 1] over dimensions 
    """
    h_max = F.max(h, axis=1, keepdims=True)
    h_min = F.min(h, axis=1, keepdims=True)
    h_norm = (h - h_min) / (h_max - h_min + 1e-10)
    
    return h_norm
        
def save_incorrect_info(x_rec, x_l, y, y_l):
    # Generated Images
    if os.path.exists("./test_gen"):
        shutil.rmtree("./test_gen")
        os.mkdir("./test_gen")
    else:
        os.mkdir("./test_gen")

    # Images
    if os.path.exists("./test"):
        shutil.rmtree("./test")
        os.mkdir("./test")
    else:
        os.mkdir("./test")
     
    # Generated Images
    for i, img in enumerate(x_rec):
        fpath = "./test_gen/{:05d}.png".format(i)
        cv2.imwrite(fpath, img.reshape(28, 28) * 255.)
     
    # Images
    for i, img in enumerate(x_l):
        fpath = "./test/{:05d}.png".format(i)
        cv2.imwrite(fpath, img.reshape(28, 28) * 255.)

    # Label and Probability
    with open("./label_prediction.out", "w") as fpout:
        header = ["idx", "true", "pred"]
        header += ["prob_{}".format(i) for i in range(len(y[0]))]
        writer = csv.writer(fpout, delimiter=",")
        writer.writerow(header)
        for i, elm in enumerate(zip(y, y_l)):
            y_, y_l_ = elm
            row = [i] + [y_l_] + [np.argmax(y_)] + map(lambda x: "{:05f}".format(x) , y_.tolist())
            writer.writerow(row)
            
