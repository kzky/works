from da_recon.experiments import Experiment006
from da_recon.utils import to_device
from da_recon.datasets import MNISTDataReader, Separator
import numpy as np
import os
import numpy as np
import sys
import time
import chainer.functions as F
from chainer import Variable


"""
#1. Intermidiate good results (Not using BatchNorm)

$ python exp006.py
Num. of labeled samples 6000
Num. of unlabeled samples 60000
Num. of test samples 10000
Num. of classes 10
# Training loop
Epoch:1,ElapsedTime:39.3988780975,Acc:0.938099980354,SupervisedLoss:0.778847634792,ReconstructionLoss:0.0920879468322
Epoch:2,ElapsedTime:38.6708779335,Acc:0.945500016212,SupervisedLoss:0.704142570496,ReconstructionLoss:0.0791729390621
Epoch:3,ElapsedTime:39.569370985,Acc:0.947600007057,SupervisedLoss:0.754851877689,ReconstructionLoss:0.0755339413881
Epoch:4,ElapsedTime:39.672962904,Acc:0.952899992466,SupervisedLoss:0.716730058193,ReconstructionLoss:0.0686608031392
Epoch:5,ElapsedTime:41.0056860447,Acc:0.94470000267,SupervisedLoss:0.777901411057,ReconstructionLoss:0.0788391679525
Epoch:6,ElapsedTime:39.6937179565,Acc:0.953199982643,SupervisedLoss:0.742242634296,ReconstructionLoss:0.0663816556334

#2. Intermidiate good results (Using BatchNorm and Noise even in test)
Depending on datasets?
/home/kzk/datasets/mnist/{l_train_good_results.npz, u_train_good_results.npz}

-> No
It did not depend on randomly sampled dataset because even if running experiments some time, the results are almost the same, 95% accuracy.

 python exp006.py
Num. of labeled samples 6000
Num. of unlabeled samples 60000
Num. of test samples 10000
Num. of classes 10
# Training loop
Epoch:1,ElapsedTime:80.6847538948,Acc:0.950900018215,SupervisedLoss:1.08128976822,ReconstructionLoss:0.136103391647
Epoch:2,ElapsedTime:79.8726809025,Acc:0.947399973869,SupervisedLoss:0.84667134285,ReconstructionLoss:0.125570893288
Epoch:3,ElapsedTime:78.6997539997,Acc:0.954400002956,SupervisedLoss:0.675140738487,ReconstructionLoss:0.121869444847
Epoch:4,ElapsedTime:78.8419821262,Acc:0.950999975204,SupervisedLoss:0.671376407146,ReconstructionLoss:0.120546206832
Epoch:5,ElapsedTime:79.7258489132,Acc:0.95450001955,SupervisedLoss:0.615385890007,ReconstructionLoss:0.121254965663
Epoch:6,ElapsedTime:79.6311910152,Acc:0.954400002956,SupervisedLoss:0.648241519928,ReconstructionLoss:0.113617710769
Epoch:7,ElapsedTime:79.8232250214,Acc:0.953800022602,

"""

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = 128
    inp_dim = 784
    out_dim = n_cls = 10
    n_l_train_data = 100
    n_train_data = 60000

    dims = [inp_dim, 250, 100, out_dim]
    lambdas = [1., 1., 1.]
    learning_rate = 1. * 1e-3
    n_epoch = 100
    decay = 0.5
    act = F.relu
    noise = False
    bn = True
    lateral = False
    test = False
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
    exp = Experiment006(
        device,
        learning_rate,
        lambdas,
        dims,
        act,
        noise,
        bn,
        lateral,
        test)

    # Training loop
    print("# Training loop")
    epoch = 1
    st = time.time()
    for i in range(n_iter):
        # Get data
        x_l, y_l = [Variable(to_device(x, device)) \
                        for x in data_reader.get_l_train_batch()]
        x_u, _ = [Variable(to_device(x, device)) \
                      for x in data_reader.get_u_train_batch()]

        # Train
        exp.train(x_l, y_l, x_u)
        
        # Eval
        if (i+1) % iter_epoch == 0:
            # Get data
            x_l, y_l = [Variable(to_device(x, device)) \
                            for x in data_reader.get_test_batch()]
            exp.test(x_l, y_l)

            acc, sloss, rloss = exp.test(x_l, y_l)
            msg = "Epoch:{},ElapsedTime:{},Acc:{},SupervisedLoss:{},ReconstructionLoss:{}".format(epoch, time.time() - st, acc.data, sloss.data, rloss.data)
            print(msg)
            
            st = time.time()
            epoch +=1
            
if __name__ == '__main__':
    main()
