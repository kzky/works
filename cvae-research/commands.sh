#!/bin/bash

screen -Sdm "exampl-0" bash -c "python train.py -d 2 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_0 --use-deconv"

# Use min/max threshold for logvar
screen -Sdm "exampl-1" bash -c "python train.py -d 2 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_1 --use-deconv"

# Use upsampling -> convolution 
screen -Sdm "exampl-2" bash -c "python train.py -d 3 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_2"


