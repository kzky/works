#!/bin/bash

# bn-000
python train.py --device-id 2 \
                --img-path ~/nnabla_data/celebA/img_align_celeba_png \
                --norm BN \
                --monitor-path ./result/bn-000
# in-000
python train.py --device-id 2 \
                --img-path ~/nnabla_data/celebA/img_align_celeba_png \
                --norm IN \
                --monitor-path ./result/in-000

# ccbn-000
python train.py --device-id 2 \
                --img-path ~/nnabla_data/celebA/img_align_celeba_png \
                --attr-path ~/nnabla_data/celebA/list_attr_celeba.csv \
                --norm CCBN \
                --monitor-path ./result/ccbn-000


# ccin-000
python train.py --device-id 3 \
                --img-path ~/nnabla_data/celebA/img_align_celeba_png \
                --attr-path ~/nnabla_data/celebA/list_attr_celeba.csv \
                --norm CCIN \
                --monitor-path ./result/ccin-000


############
# Generation
############
python generate.py --device-id 2 \
                   --model-load-path ./result/bn-000/Gen_phase_128_epoch_4.h5 \
                   --monitor-path ./result/bn-000


