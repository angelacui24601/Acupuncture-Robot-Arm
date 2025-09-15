#!/bin/bash

set -e
set -x
#
#data_dir="/data/coco2017/"
#output_dir="./output/hsnet_qt_vit_HW_coco_r50_800ep"

exp_id="AcupointMM"

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 12341 --nproc_per_node=3 \
    ../train.py \
    --dataset hand \
    --exp_id ${exp_id} \
    \
    --arch acupointmm \
    --batch_size 6 \
    --optimizer adam \
    --base-lr 1 \
    --num_epochs 80 \
    --print-freq 10 \
    --save-freq 20 \
    --auto-resume \
    --num-workers 0 \
    --load_model '../results/acupointmm/AcupointMM/current.pth' \
#    --fp16 \



