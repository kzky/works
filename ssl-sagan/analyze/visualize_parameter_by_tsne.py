import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.parameter import get_parameter_or_create

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from args import get_args, save_args

from collections import OrderedDict

def main():
    # Args
    args = get_args()
    save_args(args, mode="viz")
    
    # Model load path
    nn.load_parameters(args.model_load_path)
    embed_w = OrderedDict()
    for k, v in nn.get_parameters(grad_only=False).items():
        if "embed" in k and "singular" not in k:
            embed_w[k.replace("/", "-")] = v.d.copy()

    # t-SNE
    for k, v in embed_w.items():
        X_reduced = TSNE(n_components=2, random_state=0).fit_transform(v)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=np.arange(v.shape[0]))
        plt.title(k)
        path = "{}/tsne-{}".format(args.monitor_path, k)
        plt.savefig(path)
        plt.clf()
        print("{} ({}) is saved".format(path, v.shape))

    # Histogram
    for k, v in embed_w.items():
        plt.hist(v.flatten(), bins=200)
        plt.title(k)
        path = "{}/hist-{}".format(args.monitor_path, k)
        plt.savefig(path)
        plt.clf()
        print("{} ({}) is saved".format(path, v.shape))


if __name__ == '__main__':
    main()
    
