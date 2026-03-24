#!/bin/bash
# Experiment 1: Train on STARSS23, test on internal dataset
for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=0 python seld.py \
    -train -val \
    -b ${batch_size} \
    -s 1000 -i 20000 \
    -twt ./data_dcase2023_task3/list_dataset/dcase2023t3_foa_devtrain_audiovisual.txt \
    -valwt ./data_dcase2023_task3/list_dataset/internal_test.txt \
    -m ./data_dcase2023_task3/model_monitor/exp1_starss_train_internal_test;
done