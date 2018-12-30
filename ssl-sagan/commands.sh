#!/bin/bash

# Train command
# python train.py -d 0 -c cudnn -b 32 -a 2 -t float \
#        -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_abs \
#        -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
#        --n-classes 10 \
#        --monitor-path ./result/example_000 \
#        --max-iter 100000 \
#        --save-interval 1000


# python train.py -d 1 -c cudnn -b 32 -a 1 -t float \
#        -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_abs \
#        -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
#        --monitor-path ./result/example_001 \
#        --n-classes 10 \
#        --max-iter 200000 \
#        --save-interval 2000

#####################
# 10 picked clasees
#####################
python train.py -d 0 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_000 \
       --n-classes 10 \
       --max-iter 10000 \
       --save-interval 100

python train.py -d 1 -c cudnn -b 32 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_001 \
       --n-classes 10 \
       --max-iter 80000 \
       --save-interval 800

# Fix iteration with large batch size -> bug in loss scaling using accum_grad
python train.py -d 0 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_002 \
       --n-classes 10 \
       --max-iter 100000 \
       --save-interval 1000

# Fix iteration with small batch size 
python train.py -d 1 -c cudnn -b 32 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_003 \
       --n-classes 10 \
       --max-iter 800000 \
       --save-interval 8000

# 004: Fix iteration with large batch size and bug in accum_grad
python train.py -d 0 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_004 \
       --n-classes 10 \
       --max-iter 100000 \
       --save-interval 1000

# 005: Fix iteration with large batch size, fix bug in accum_grad, small model
python train.py -d 1 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_005 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 100000 \
       --save-interval 1000

# 006: Fix iteration with large batch size, fix bug in accum_grad, small model, flip
python train.py -d 0 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_006 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 100000 \
       --save-interval 1000 \
       --flip

# 007: Fix iteration with large batch size, fix bug in accum_grad, 2x small model, flip
python train.py -d 1 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_007 \
       --n-classes 10 \
       --maps 256 \
       --max-iter 100000 \
       --save-interval 1000 \
       --flip


# 008: Fix iteration with large batch size, fix bug in accum_grad, small model, flip
mpirun -n 4 python train_with_mgpu.py -d 0 -c cudnn -b 64 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_008 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 250000 \
       --save-interval 1000 \
       --flip

#######################
# 10 picked dog classes
#######################
python train.py -d 2 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_000 \
       --n-classes 10 \
       --max-iter 10000 \
       --save-interval 100


python train.py -d 3 -c cudnn -b 32 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_001 \
       --n-classes 10 \
       --max-iter 80000 \
       --save-interval 800

# Fix iteration with large batch size -> bug in loss scaling using accum_grad
python train.py -d 2 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_002 \
       --n-classes 10 \
       --max-iter 100000 \
       --save-interval 1000

# Fix iteration with small batch size
python train.py -d 3 -c cudnn -b 32 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_003 \
       --n-classes 10 \
       --max-iter 800000 \
       --save-interval 8000

# 004: Fix iteration with large batch size, fix bug in accum_grad
python train.py -d 2 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_004 \
       --n-classes 10 \
       --max-iter 100000 \
       --save-interval 1000

# 005: Fix iteration with large batch size, fix bug in accum_grad, small model
python train.py -d 3 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_005 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 100000 \
       --save-interval 1000

# 006: Fix iteration with large batch size, fix bug in accum_grad, small model
python train.py -d 2 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_006 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 100000 \
       --save-interval 1000 \
       --flip

# 007: Fix iteration with large batch size, fix bug in accum_grad, small model
python train.py -d 3 -c cudnn -b 32 -a 8 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10_dogs \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_dogs_abs.txt \
       --monitor-path ./result/10_picked_dog_007 \
       --n-classes 10 \
       --maps 256 \
       --max-iter 100000 \
       --save-interval 1000 \
       --flip

#######################
# dog or cat
#######################

mpirun -n 4 python train_with_mgpu.py -d 0 -c cudnn -b 64 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_dog_or_cat \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_dog_or_cat_abs.txt \
       --monitor-path ./result/dog_or_cat_000 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 450000 \
       --save-interval 10000 \
       --flip

#######################
# dog
#######################
mpirun -n 4 python train_with_mgpu.py -d 0 -c cudnn -b 64 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_dog \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_dog_abs.txt \
       --monitor-path ./result/dog_000 \
       --n-classes 10 \
       --maps 512 \
       --max-iter 450000 \
       --save-interval 10000 \
       --flip
