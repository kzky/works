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
python train.py -d 0 -c cudnn -b 32 -a 2 -t float \
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
       --max-iter 20000 \
       --save-interval 200


#######################
# 10 picked dog classes
#######################
python train.py -d 2 -c cudnn -b 32 -a 2 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_dog_000 \
       --n-classes 10 \
       --max-iter 10000 \
       --save-interval 100


python train.py -d 3 -c cudnn -b 32 -a 1 -t float \
       -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan_picked_10 \
       -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label_picked_10_abs.txt \
       --monitor-path ./result/10_picked_dog_001 \
       --n-classes 10 \
       --max-iter 20000 \
       --save-interval 200
