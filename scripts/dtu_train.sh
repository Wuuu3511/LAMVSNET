#!/usr/bin/env bash
datapath="you dtu training dataset path"
log_dir="./ckpt"
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=4239 main.py \
        --sync_bn \
        --view_shuffling \
        --ndepths 48 32 8 \
        --interval_ratio 4 1 0.5\
        --img_size 512 640 \
        --train_view 5 \
        --dlossw 0.5 1.0 2.0 \
        --log_dir $log_dir \
        --datapath $datapath \
        --dataset_name "dtu_yao" \
        --epochs 16 \
        --batch_size 4 \
        --lr 0.001 \
        --warmup 0.2 \
        --scheduler "steplr" \
        --milestones 10 12 14 \
        --lr_decay 0.5 \
        --trainlist "datasets/lists/dtu/train.txt" \
        --testlist "datasets/lists/dtu/test.txt" \
        --numdepth 192 \
        --interval_scale 1.06 ${@:1} | tee -a $log_dir/log.txt > train.log 2>&1 &