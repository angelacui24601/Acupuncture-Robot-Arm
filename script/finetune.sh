#!/bin/bash

set -e
set -x
#
#data_dir="/data/coco2017/"
#output_dir="./output/hsnet_qt_vit_HW_coco_r50_800ep"

#exp_id="hsnet_qt_all_rgb_mask_CA_bs256_finetune"
exp_id="AcupointMM"
#12345
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 12340 --nproc_per_node=3 \
    ../finetune.py \
    --dataset hand \
    --exp_id ${exp_id} \
    \
    --arch acupointmm \
    \
    --batch_size 3 \
    --optimizer adam \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --num_epochs 200 \
    --print-freq 10 \
    --save-freq 20 \
    --auto-resume \
    --num-workers 4 \
#    --load_model '/data02/DockerComparisons/RelatedSources/SourceCodes/ZhangZhanCode/results/hsnet_qt_all_rgb_mask_CA_D12/current.pth'


