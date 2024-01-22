#!/usr/bin/env bash
datapath="your tnt dataset path"
outdir="./output_tnt"
resume="./ckpt/bld.ckpt"
CUDA_VISIBLE_DEVICES=0 python main.py \
        --test \
        --ndepths 96 64 16 \
        --interval_ratio 2 0.5 0.25 \
        --train_view 9 \
        --test_view 11 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "all" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "dypcd" ${@:1}