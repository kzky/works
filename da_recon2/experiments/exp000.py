from da_recon2.experiments import Experiment000
from da_recon2.utils import to_device, to_onehot
from da_recon2.datasets import MNISTDataReader, Separator
import numpy as np
import os
import numpy as np
import sys
import time
import chainer.functions as F
from chainer import Variable
from chainer import serializers
import cv2

a = ["MLP", "CNN"]
b = ["No", "Noisy"]
c = ["Gen-Enc", "Gen-Dec", "Gen-Enc/Gen-Dec"]
d = ["Gen-Enc", "Gen-Dec", "Gen-Enc/Gen-Dec"]
e = ["No", "D(x)", "D(x_recon)", "Both"]

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = 128
    out_dim = n_cls = 10
    n_l_train_data = 100
    n_train_data = 60000

    learning_rate = 1. * 1e-3
    n_epoch = 100
    decay = 0.5
    act = F.relu
    sigma = 0.3
    ncls = n_cls
    dim = 100
    noise = False
    rc_feet = "Gen-Enc"
    rc_sample = "Gen-Enc"
    gan_loss = "No"
    
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch

    # Separate dataset
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/mnist/train.npz")
    separator = Separator(n_l_train_data)
    separator.separate_then_save(fpath)

    l_train_path = os.path.join(home, "datasets/mnist/l_train.npz")
    u_train_path = os.path.join(home, "datasets/mnist/train.npz")
    test_path = os.path.join(home, "datasets/mnist/test.npz")

    # DataReader, Model, Optimizer, Losses
    data_reader = MNISTDataReader(l_train_path, u_train_path, test_path,
                                  batch_size=batch_size,
                                  n_cls=n_cls)
    exp = Experiment000(
        device,
        learning_rate,
        act,
        sigma,
        n_cls,
        dim,
        noise,
        rc_feet,
        rc_sample,
        gan_loss)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    utime = int(st)
    dpath0 = "./{}_{}".format(os.path.basename(__file__), utime)
    os.mkdir(dpath0)
    v_losses = []
    recon_losses = []
    recon_feet_losses = []
    for i in range(n_iter):
        # Get data
        x_l_, y_l_ = data_reader.get_l_train_batch()
        x_l = Variable(to_device(x_l_, device))
        y_l = Variable(to_device(to_onehot(y_l_, ncls), device))
        x_u, _ = [Variable(to_device(x, device)) \
                      for x in data_reader.get_u_train_batch()]

        # Train
        exp.train(x_l, y_l, x_u)
        v_losses.append(exp.v_loss_data)
        recon_losses.append(exp.recon_loss_data)
        recon_feet_losses.append(exp.recon_feet_loss_data)
        
        # Eval
        if (i + 1) % iter_epoch == 0:
            v_loss = np.mean(v_losses)
            recon_loss = np.mean(recon_losses)
            recon_feet_loss = np.mean(recon_feet_losses)
            msg = "Epoch{:03d},VLoss:{},RLoss:{},RFLoss:{}".format(
                epoch, v_loss, recon_loss, recon_feet_loss)
            print(msg)
            v_losses = []
            recon_losses = []
            recon_feet_losses = []
            
            bs = 16
            # Generate
            ## for unlabel
            x_u_ = exp.generate(bs, dim)
            x_u = (to_device(x_u_.data, None) * 255.)\
                .astype(np.int32)\
                .reshape(bs, 28, 28)

            ## for label
            y_ = np.random.choice(n_cls, bs)
            y = to_onehot(y_, ncls)
            x_l_ = exp.generate(bs, dim)
            x_l = (to_device(x_l_.data, None) * 255.)\
                .astype(np.int32)\
                .reshape(bs, 28, 28)
            
            # Save images
            dpath1 = os.path.join(dpath0, "{:05d}".format(epoch))
            os.mkdir(dpath1)
            for i, img in enumerate(x_u):
                fpath = os.path.join(dpath1, "{:05}_u.png".format(i))
                cv2.imwrite(fpath, img)
            for i, img in enumerate(x_l):
                fpath = os.path.join(dpath1, "{:05}_l_{:03}.png".format(i, y_[i]))
                cv2.imwrite(fpath, img)
                
            # Save model
            fpath = os.path.join(dpath0, "DARECON_{:05d}.model".format(epoch))
            serializers.save_hdf5(fpath, exp.model)
            
            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    main()
