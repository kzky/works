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
    
