#!/bin/bash
# use the customised task
TASK=${1:udpos}

# (german spanish french chinese japanese)
declare -a list_of_adapt_trn_languages=(german)
# (1e-5 3e-5 5e-5 7e-5)
declare -a list_of_adapt_lr=(1e-5)
# list of number of shots depends on the dataset, 1
declare -a list_of_num_shots=(1, 2, 4)
declare -a ckpt_path=data/checkpoint_adapt
# group: bucket; we sampled 40 buckets for each target
declare -a list_of_group_index=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39)

for ((which_adapt_lang=0;which_adapt_lang<${#list_of_adapt_trn_languages[@]};++which_adapt_lang)); do
    for ((which_adapt_lr=0;which_adapt_lr<${#list_of_adapt_lr[@]};++which_adapt_lr)); do
        for ((which_num_shots=0;which_num_shots<${#list_of_num_shots[@]};++which_num_shots)); do
            for ((which_group_index=0;which_group_index<${#list_of_group_index[@]};++which_group_index)); do
                python adapt_training.py 
                    --override False \
                    --experiment udpos_adapt_1_shot_no0s \
                    --ptl bert \
                    --model bert-base-multilingual-cased \
                    --dataset_name $TASK\
                    --adapt_trn_languages ${list_of_adapt_trn_languages[which_adapt_lang]} \
                    --adapt_epochs 10 \
                    --early_stop True \
                    --early_stop_patience 10 \
                    --adapt_batch_size 32 \
                    --adapt_lr ${list_of_adapt_lr[which_adapt_lr]} \
                    --adapt_num_shots ${list_of_num_shots[which_num_shots]} \
                    --group_index ${list_of_group_index[which_group_index]} \
                    --load_ckpt False \
                    --ckpt_path ckpt_path \
                    --manual_seed 42 \
                    --train_fast True \
                    --world 0
            done
        done
    done
done

