import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker

def save_grad_scale_over_time(dpath, filter_="conv_W"):
    # Save gradient as distribution
    fpaths = glob.glob(os.path.join(dpath, "*.npz"))
    fpaths.sort()
    scale_over_time_for_sr = []
    scale_over_time_for_ce = []
    for fpath in fpaths:
        print(fpath)
        data = np.load(fpath)["arr_0"]
        data = data.reshape((np.prod(data.shape), ))
        scale = np.std(data)

        basename = os.path.basename(fpath)
        if filter_ in basename: 
            if "sr__" in basename:
                scale_over_time_for_sr.append(scale)
            if "ce__" in basename:
                scale_over_time_for_ce.append(scale)
        
    # Save figure of scale over time
    basedir = os.path.dirname(dpath)
    basename = os.path.basename(dpath)
    save_fpath = os.path.join(
        basedir, 
        "{}_{}_scale_dist_over_time.png".format(basename, filter_))
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    legend_ce, = ax.plot(scale_over_time_for_ce, 
                         label="Grad scale for CE loss")
    legend_sr, = ax.plot(scale_over_time_for_sr, 
                         label="Grad scale for SR loss")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
        
    fig.savefig(save_fpath)
    plt.close(fig)

def main():
    dpath = "/home/kzky/works/st/experiments/cifar10/1498905497_conv23"
    filter_ = "bn_gamma"
    save_grad_scale_over_time(dpath, filter_)
    
if __name__ == '__main__':
    main()
    
