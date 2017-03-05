import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv

def compute_acc_std(filepath,  epochs = 50):
    filepaths = glob.glob(os.path.join(filepath, "*"))
    trials = len(filepaths)

    accs_over_epoch = np.zeros((epochs, trials))
    for i, fpath in enumerate(filepaths):
        j = 0
        with open(fpath, "r") as fp:
            reader = csv.reader(fp, delimiter=",")
            for l in reader:
                if not l[0].startswith("Epoch"):
                    continue
                acc = l[-1].split(":")[-1].split("|")[-1]  # Use the last layers accuracy
                accs_over_epoch[j, i] = float(acc)
                j += 1
    max_values = np.max(accs_over_epoch, axis=0)
    mean = np.mean(max_values)
    std = np.std(max_values)
    return mean, std

def summarize_acc_std(filepaths, epochs = 50):

    accs_stds = []
    for fpath in filepaths:
        mean, std = compute_acc_std(fpath, epochs)
        accs_stds.append([mean, std])
        trial_name = os.path.basename(fpath)
        print("{},{},{}".format(trial_name, mean, std))
    

def main():

    filepaths = [
        "/home/kzk/documents/fy16/20160221/exp000",
        "/home/kzk/documents/fy16/20160221/exp001",
        "/home/kzk/documents/fy16/20160221/exp002",
        "/home/kzk/documents/fy16/20160221/exp008",
        "/home/kzk/documents/fy16/20160221/exp012",
        "/home/kzk/documents/fy16/20160221/exp014",
        "/home/kzk/documents/fy16/20160221/exp015",
        "/home/kzk/documents/fy16/20160221/exp022",
        "/home/kzk/documents/fy16/20160221/exp023",
    ]
    summarize_acc_std(filepaths)


if __name__ == '__main__':
    main()
