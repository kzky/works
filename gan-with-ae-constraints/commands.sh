#!/bin/bash

screen -Sdm "example-0" bash -c "python train.py -d 2 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_0 --use-deconv"

