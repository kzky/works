import numpy as np
import cv2
from chainer import serializers
from da_recon7.models import MLPEncDecModel
import chainer.functions as F

def main():
    # Settings
    inp_dim = 784
    out_dim = 10
    dims = [inp_dim, 500, 250, 100, out_dim]
    act = F.relu
    noise = False
    rc = True
    device = None

    # Model
    model = MLPEncDecModel(
        dims=dims, act=act,
        noise=noise, rc=rc,
        device=device)
    decoder = model.mlp_dec
    
    # Load
    fpath = "/home/kzk/tmp/exp090/mlp_encdec.h5py"
    model = serializers.load_hdf5(fpath, model)

    # Generate random vector(s)
    bs = 1
    y = np.random.rand(bs, 10).astype(np.float32)
    y = np.array([[1, 100, 1, 1, 500, 0, 10, 10, 10, 100]], dtype=np.float32)
    y = y / np.sum(y)
    y = y[np.newaxis]

    # Generate sample(s)
    x = decoder(y, test=True)
    x = x.data.reshape((bs, 28, 28)) * 255. 
    for i in range(bs):
        cv2.imwrite("./gen_mnist_{:05d}.png".format(i), x[i, ])

if __name__ == '__main__':
    main()
