#!/usr/bin/env bash
datapath="your bld dataset path"
resume="dtu model path"
log_dir="./ckptbld"
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=2110 main.py \
        --sync_bn \
        --view_shuffling \
        --blendedmvs_finetune \
        --resume $resume \
        --ndepths 96 64 16 \
        --interval_ratio 2 0.5 0.25 \
        --img_size 576 768 \
        --dlossw 0.5 1.0 2.0 \
        --log_dir $log_dir \
        --datapath $datapath \
        --dataset_name "blendedmvs" \
        --train_view 9 \
        --epochs 10 \
        --batch_size 1 \
        --lr 0.0001 \
        --scheduler steplr \
        --warmup 0.2 \
        --milestones 6 8 \
        --lr_decay 0.5 \
        --trainlist "datasets/lists/blendedmvs/training_list.txt" \
        --testlist "datasets/lists/blendedmvs/validation_list.txt" \
        --numdepth 128 \
        --interval_scale 1.06 ${@:1} | tee -a $log_dir/log.txt > trainbl.log 2>&1 &
