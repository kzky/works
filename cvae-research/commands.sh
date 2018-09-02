#!/bin/bash

screen -Sdm "example-0" bash -c "python train.py -d 2 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_0 --use-deconv"

# Use min/max threshold for logvar
screen -Sdm "example-1" bash -c "python train.py -d 2 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_1 --use-deconv"

# Use upsampling -> convolution 
screen -Sdm "example-2" bash -c "python train.py -d 3 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_2"


# Use upsampling -> convolution, use fft power loss (normalized) -> Wrong!
screen -Sdm "example-3" bash -c "python train.py -d 0 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_3"

screen -Sdm "example-4" bash -c "python train.py -d 0 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_4 --lam-fft 0.1"


# Use upsampling -> convolution, use fft power loss (normalized)
screen -Sdm "example-5" bash -c "python train.py -d 0 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_5 --lam-fft 1e-2"
screen -Sdm "example-6" bash -c "python train.py -d 1 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_6 --lam-fft 1e-2 --use-patch"


# Use edge loss (laplacian pyramid)
screen -Sdm "example-7" bash -c "python train.py -d 0 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_7"


# Use upsampling -> convolution w/ beta1 0.5
screen -Sdm "example-8" bash -c "python train.py -d 3 --train-data-path /data/datasets/celebA/img_align_celeba_png --monitor-path result/example_8 --beta1 0.5 --beta2 0.9"
