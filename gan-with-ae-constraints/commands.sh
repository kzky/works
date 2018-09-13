#!/bin/bash

screen -Sdm "example-0" bash -c "python train.py -d 0 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_0"


# Add weight to objectves
screen -Sdm "example-1" bash -c "python train.py -d 1 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_1"


# Add weight to objectves
screen -Sdm "example-2" bash -c "python train.py -d 0 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_2 --lam 10.0"

screen -Sdm "example-3" bash -c "python train.py -d 1 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_3 --lam 0.1"


# 
screen -Sdm "example-4" bash -c "python train.py -d 1 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_4 --use-pfvn"

# 
screen -Sdm "example-5" bash -c "python train.py -d 2 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_5"



