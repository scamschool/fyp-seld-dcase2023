#!/bin/bash
# Experiment 2: Finetune on internal dataset, test on internal dataset
# NOTE: Replace <TIMESTAMP> and <ITERATION> with actual values from Exp 1 output
# e.g. params_20250101120000_0020000.pth
EXP1_MODEL="./data_dcase2023_task3/model_monitor/exp1_starss_train_internal_test/<TIMESTAMP>/params_<TIMESTAMP>_0020000.pth"

for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=0 python seld.py \
    -train -val \
    -b ${batch_size} \
    -s 1000 -i 20000 \
    -twt ./data_dcase2023_task3/list_dataset/internal_train.txt \
    -valwt ./data_dcase2023_task3/list_dataset/internal_test.txt \
    -m ./data_dcase2023_task3/model_monitor/exp2_finetune_internal_test \
    -pm ${EXP1_MODEL};
done