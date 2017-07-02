import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker

def save_grad_dist_over_time(dpath):
    # Create savedir
    basedir = os.path.dirname(dpath)
    basename = os.path.basename(dpath)
    save_dpath = os.path.join(basedir, "{}_dist".format(basename))
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    # Save gradient as distribution
    fpaths = glob.glob(os.path.join(dpath, "*.npz"))
    fpaths.sort()
    for fpath in fpaths:
        print(fpath)
        data = np.load(fpath)["arr_0"]
        data = data.reshape((np.prod(data.shape), ))
        scale = np.std(data)
        fig, ax = plt.subplots()
        ax.hist(data, bins=100)
        ax.set_title("Gradient Histrogrm ($\sigma$={:05f})".format(scale))
        ax.set_xlabel("Gradient ($ \\times 10^{-3}$)")
        ax.set_ylabel("Frequency")
        #for label in ax.get_xticklabels():
        #    label.set_rotation(-30)
        scale_x = 1e-3
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        ax.xaxis.set_major_formatter(ticks_x)

        basename = os.path.basename(fpath)
        filename, ext = os.path.splitext(basename)
        save_fpath = os.path.join(save_dpath, "{}.png".format(filename))
        fig.savefig(save_fpath)
        plt.close(fig)

def main():
    dpath = "/home/kzk/project/ssl/stochastic_regularization/results/"\
            "1498905497_conv02"
    save_grad_dist_over_time(dpath)

    dpath = "/home/kzk/project/ssl/stochastic_regularization/results/"\
            "1498905497_conv12"
    save_grad_dist_over_time(dpath)

    dpath = "/home/kzk/project/ssl/stochastic_regularization/results/"\
            "1498905497_conv23"
    save_grad_dist_over_time(dpath)
    
    
if __name__ == '__main__':
    main()
    
