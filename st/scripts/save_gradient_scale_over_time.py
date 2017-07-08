import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker

def save_grad_scale_over_time(dpath):
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
        if "sr__" in basename:
            scale_over_time_for_sr.append(scale)
        if "ce__" in basename:
            scale_over_time_for_sr.append(scale)
        
    # Save figure of scale over time
    scale_over_time = np.concatenate(
        scale_over_time_for_sr, 
        scale_over_time_for_ce)
    basedir = os.path.dirname(dpath)
    basename = os.path.basename(dpath)
    save_fpath = os.path.join(basedir, 
                              "{}_scale_dist_over_time.png".format(basename))
    fig, ax = plt.subplots()
    ax.plot(scale_over_time)
    fig.save(save_fpath)
    plt.close(fig)

def main():
    dpath = "/home/kzk/project/ssl/stochastic_regularization/results/"\
            "1498905497_conv02"
    save_grad_scale_over_time(dpath)
    
if __name__ == '__main__':
    main()
    
