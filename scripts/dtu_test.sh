#!/usr/bin/env bash
datapath="your dtu testing dataset path"
outdir="./outputsdtu"
resume="./ckpt/dtu.ckpt"
fusibile_exe_path="./fusibile"
CUDA_VISIBLE_DEVICES=1 python main.py \
        --test \
        --ndepths 48 32 8 \
        --interval_ratio 4 1 0.5 \
        --max_h 864 \
        --max_w 1152 \
        --train_view 5 \
        --test_view 5 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "datasets/lists/dtu/test.txt" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "dypcd_dtu" \
        --fusibile_exe_path $fusibile_exe_path \
        --prob_threshold 0.01 \
        --disp_threshold 0.25 \
        --num_consistent 3 ${@:1}
