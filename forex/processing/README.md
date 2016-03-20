# Interpolation of data having missing valued

## Data 

Data downloaded in [here](http://www.histdata.com/download-free-forex-data/)

# CSV Format

|Date|Time(1min)|Bar OPEN Bid Quote|Bar HIGH Bid Quote|Bar LOW Bid Quote|Bar CLOSE Bid Quote|Volume|
|2003.01.01|19:14|124.340000|124.380000|124.320000|124.370000|0|
|2003.01.01|19:15|124.390000|124.460000|124.360000|124.400000|0|
|2003.01.01|19:16|124.390000|124.470000|124.370000|124.460000|0|
|2003.01.01|19:17|124.480000|124.480000|124.400000|124.410000|0|

## Gap data format

There are no format. The following is data excerpt.

```
HistData.com (c) 2012
File: DAT_MT_EURJPY_M1_2003.csv Status Report

Gap of 68s found between 20030101190024 and 20030101190138.
Gap of 117s found between 20030101190707 and 20030101190910.
Gap of 129s found between 20030101191049 and 20030101191304.
Gap of 135s found between 20030101192040 and 20030101192301.
Gap of 142s found between 20030101192342 and 20030101192610.
Gap of 66s found between 20030101192610 and 20030101192722.
```

## Interpolation Logic
- load gapped data into memory
- load csv data into memroy
- create new data bucket
- loop csv data
-- take time difference between at t-1 and t
-- if differenece is greater than 1, then see the gapped data
--- if time data is in gapped data, interpolate data using the previous data among gapped period
-- insert these into data bucket
- save


# Extract records from interpolated dataset

If specifying n-min unit (e.g., 10min, 15min, 30min , 60min, 90min, 120min), then extract the n-min-unit records from interpolated dataset.
