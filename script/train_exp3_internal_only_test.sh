#!/bin/bash
# Experiment 3: Train on internal dataset only, test on internal dataset
for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=0 python seld.py \
    -train -val \
    -b ${batch_size} \
    -s 1000 -i 20000 \
    -twt ./data_dcase2023_task3/list_dataset/internal_train.txt \
    -valwt ./data_dcase2023_task3/list_dataset/internal_test.txt \
    -m ./data_dcase2023_task3/model_monitor/exp3_internal_only_test;
done