#!/usr/bin/env python

import numpy as np
import csv

from filters import compute_rkzfilter

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
    fvalues_m_t_36 = rkz[36:, m]

    header = ["Date Time", "Close Price", "RKZ (m)", "RKZ (m) at t - 36"]
    with open(filepath, "w") as fpout:
            writer = csv.writer(fpout, delimiter=",")
            writer.writerow(header)
            for d, t, v0, fv_m, fv_m_t_36 in zip(date_time[:, 0], date_time[:, 1], values,
                                                 fvalues_m, fvalues_m_t_36):
                writer.writerow(["{} {}".format(d.replace(".", "/"), t), v0, fv_m, fv_m_t_36])
    
if __name__ == '__main__':
    """
    Sample
    
    python main/save_rkzfilter.py
    """
    main()
