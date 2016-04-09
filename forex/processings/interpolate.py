#!/usr/bin/env python

import numpy as np
import os
import csv
import datetime
import glob

class Interpolator(object):
    """Some values in the original dataest are missing, or gapping.
    """
    DIR_PREFIX = "interpolated"
     
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

    def interpolate_then_save(self, data_filepath, gdata_filepath):
        """Interpolate the data with gapped information and save the result.

        Saved dirpath is the basedir of the data_filepath with the prefix,
        'interpolated' and the file will be saved in the same directory level of the basedir of the data_filepath.

        Parameters
        -----------------
        
        data_filepath: str
            Filepath to data, the csv file corresponding to 1 year data.
            This data must have to correspond to gdata_filepath.

        gdata_filepath: str
            Filepath to gapped information.
            This data must have to correspond to gdata_filepath.

        """
        # Load datas
        data = np.loadtxt(data_filepath, dtype=np.str, delimiter=",")
        gdata = self._load_and_convert_gdata(gdata_filepath)
     
        # Saved data
        interpolated_data = []
     
        # Interpolates
        for p_record, c_record in zip(data[0:], data[1:]):
            # add data
            interpolated_data.append(list(p_record))
     
            # calc index diff
            prev_datetime = self._to_datetime(p_record)
            prev_time = "{},{}".format(p_record[self.DATE], p_record[self.TIME])
            index_diff = self._calc_index_diff(p_record, c_record)
     
            if index_diff > 1:
                if prev_time in gdata:
                    # interpolate among index differences
                    for i in xrange(1, index_diff):
                        print prev_time
     
                        # add 1-min
                        time = prev_datetime + datetime.timedelta(seconds=60 * i)
                        record = [
                            "{:04}.{:02}.{:02}".format(time.year, time.month, time.day),
                            "{:02}:{:02}".format(time.hour, time.minute),
                            p_record[self.OPEN_PRICE], p_record[self.HIGH_PRICE],
                            p_record[self.LOW_PRICE], p_record[self.CLOSE_PRICE],
                            p_record[self.VOLUME]
                        ]
     
                        # interpolate by 1min
                        interpolated_data.append(record)
        # add the last line
        interpolated_data.append(data[-1])
        
        # Save
        original_dirpath = os.path.dirname(data_filepath)
        dirpath = "{}/{}_{}".format(
            "/".join(original_dirpath.split("/")[0:-1]),
            self.DIR_PREFIX,
            original_dirpath.split("/")[-1])
        filename = os.path.basename(data_filepath)
        interpolated_data_filepath = "{}/{}".format(dirpath, filename)
     
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        if os.path.exists(interpolated_data_filepath):
            os.remove(interpolated_data_filepath)
     
        with open(interpolated_data_filepath, "w") as fpout:
            writer = csv.writer(fpout)
            for l in interpolated_data:
                writer.writerow(l)
     
    def _calc_index_diff(self, prev_data, curr_data):
     
        # previous time
        prev_time = self._to_datetime(prev_data)
        
        # current time
        curr_time = self._to_datetime(curr_data)
     
        td = curr_time - prev_time
        index_diff = td.seconds / 60
     
        return index_diff
     
    def _to_datetime(self, data):
        year, month, day = data[self.DATE].split(".")
        hour, minute = data[self.TIME].split(":")
        second = "00"
     
        time = datetime.datetime(int(year), int(month), int(day),
                                 int(hour), int(minute), int(second))
     
        return time
                
    def _load_and_convert_gdata(self, gdata_filepath):
        gdata = {}
        with open(gdata_filepath) as fpin:
            for l in fpin:
                if l.startswith("Gap"):
                    l0 = l.strip(".")
                    elm = l0.split(" ")[5]
                    gap_started = "{}.{}.{},{}:{}".format(elm[0:4],
                                                          elm[4:6],
                                                          elm[6:8],
                                                          elm[8:10],
                                                          elm[10:12])
                    gdata[gap_started] = 1
     
        return gdata
    
def main():

    base_dirpaths = [
        "/home/kzk/datasets/forex/EURJPY",
        "/home/kzk/datasets/forex/USDJPY",
        "/home/kzk/datasets/forex/GBPJPY",
        "/home/kzk/datasets/forex/GBPUSD",
        "/home/kzk/datasets/forex/EURUSD",
    ]

    for base_dirpath in base_dirpaths:
        
        data_filepaths = sorted(glob.glob("{}/*.csv".format(base_dirpath)))
        gdata_filepaths = sorted(glob.glob("{}/*.txt".format(base_dirpath)))

        for data_filepath, gdata_filepath in zip(data_filepaths, gdata_filepaths):
            print "--------------------"
            print data_filepath, gdata_filepath
            
            interpolator = Interpolator()
            interpolator.interpolate_then_save(data_filepath, gdata_filepath)
            
            print "--------------------"
            
if __name__ == '__main__':
    main()
