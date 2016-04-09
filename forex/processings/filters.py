#!/usr/bin/env python

import numpy as np

def compute_rkzfilter(values, n=6, m=6):
    """

    Parameters
    ---------------
    values: 1d-array
        numpy 1d-array
    
    n: int
        filter length
    
    m: int
        number of filter applications
    
    Returns
    ------------
    rkz: 2d-array
        2d-array, 0-d shows time and 1-d shows the order of m + 1,
        0-th represents the original value.
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
    filepath = "/home/kzk/datasets/forex/interpolated_USDJPY_30/DAT_MT_USDJPY_M1_2005.csv"
    
    values = np.loadtxt(filepath, usecols=(5, ), delimiter=",")
    rkz = compute_rkzfilter(values, n=6, m=6)

    print rkz
    
    
if __name__ == '__main__':
    main()
