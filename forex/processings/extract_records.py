#!/usr/bin/env python

import numpy as np
import os
import csv
import datetime
import glob


class RecordsExtractor(object):
    """Extract Records in the uni of min_unit, then save.
    """
     
    DATE = 0
    TIME = 1
    OPEN_PRICE = 2
    HIGH_PRICE = 3
    LOW_PRICE = 4
    CLOSE_PRICE = 5
    VOLUME = 6

    def __init__(self, ):
        """
        """
            
    def extract_then_save(self, data_filepath, min_unit=30):
        """Extract the records in the unit of min_unit and save.

        Saved dirpath is the basedir of data_filepath with the siffux, 'min_unit' and the file will be saved in the same directory level of the basedir of the data_filepath.

        :param data_filepath: Filepath to data, the interpolated csv file corresponding to 1 year data.
        :type data_filepath: str
        :param min_unit: the unit of minutes
        :type min_unit: int
        """

        # Load data
        data = np.loadtxt(data_filepath, dtype=np.str, delimiter=",")
        extracted_data = []

        for record in data:
            t = record[self.TIME]
            h, m = t.split(":")
            mins_in_day = int(h) * 60 + int(m)
            if mins_in_day % min_unit == 0:
                print t
                extracted_data.append(list(record))
                
        # Save
        original_dirpath = os.path.dirname(data_filepath)
        suffix = min_unit
        dirpath = "{}/{}_{}".format(
            "/".join(original_dirpath.split("/")[0:-1]),
            original_dirpath.split("/")[-1],
            suffix)
        filename = os.path.basename(data_filepath)
        extracted_data_filepath = "{}/{}".format(dirpath, filename)
     
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        if os.path.exists(extracted_data_filepath):
            os.remove(extracted_data_filepath)

        np.savetxt(extracted_data_filepath, extracted_data,
                   fmt="%s", delimiter=",")
            
def main():
    base_dirpaths = [
        "/home/kzk/downloads/interpolated_EURJPY",
        "/home/kzk/downloads/interpolated_USDJPY",
        "/home/kzk/downloads/interpolated_GBPJPY",
        "/home/kzk/downloads/interpolated_GBPUSD",
        "/home/kzk/downloads/interpolated_EURUSD",
    ]

    for base_dirpath in base_dirpaths:
        
        data_filepaths = sorted(glob.glob("{}/*.csv".format(base_dirpath)))

        for data_filepath in data_filepaths:
            print "--------------------"
            print data_filepath
            
            extractor = RecordsExtractor()
            extractor.extract_then_save(data_filepath, min_unit=60)
            
            print "--------------------"

if __name__ == '__main__':
    main()
        
