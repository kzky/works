#!/usr/bin/env python
import csv
import numpy as np

def compute_rkzfilter(values, n=6, m=6):
    """

    Parameters
    ---------------
    values: 1d-array
        Numpy 1d-array. It represents values over time.
    
    n: int
        Filter length
    
    m: int
        Number of filter applications
    
    Returns
    ------------
    rkz: 2d-array
        2d-array, 0-th axis shows time and 1st axis shows the order of m + 1,
        0-th at 2nd axis represents the original value over time.
    """

    T = len(values)
    alpha = np.round((n + 1 / 2), 0)
    rkz = np.zeros((T, m + 1))
    rkz[:, 0] = values
    
    for i in xrange(0, m):
        values_ = rkz[:, i]  # copy
        for t in xrange(alpha + i - 1, T):
            values_[t] = np.sum(values_[(t - alpha + 1):(t + 1)]) / alpha
        rkz[:, i + 1] = values_
    return rkz
    
def main():
    # Config
    currency_pair = "USDJPY"
    min_unit = 30
    n = 6
    m = 6

    # Read data
    filepath = "/home/kzk/datasets/forex/interpolated_{}_{}/DAT_MT_USDJPY_M1_2005.csv".format(currency_pair, min_unit)
    date_time = np.loadtxt(filepath, dtype="S10", usecols=(0, 1), delimiter=",")
    values = np.loadtxt(filepath, usecols=(5, ), delimiter=",")

    # Apply filter
    rkz = compute_rkzfilter(values, n=n, m=m)

    # Save i-th convolved data as csv
    filepath = "/var/www/html/data/{}_{}_{}.csv".format(currency_pair, min_unit, m)
    
    fvalues_m = rkz[:, m]

    header = ["Date Time", "Close Price", "RKZ (m)"]
    with open(filepath, "w") as fpout:
            writer = csv.writer(fpout, delimiter=",")
            writer.writerow(header)
            for d, t, v0, fv_m in zip(date_time[:, 0], date_time[:, 1], values,
                                      fvalues_m):
                writer.writerow(["{} {}".format(d.replace(".", "/"), t), v0, fv_m])
    
if __name__ == '__main__':
    main()
