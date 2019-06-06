#!/bin/bash

CONFIG_FILE=$1
if [ $CONFIG_FILE ]; then
    echo 'Using config file', $CONFIG_FILE
else
    echo 'Please give a config file as the first argument'
    exit 1
fi

CUDA_DEVICES=$2
if [ $CUDA_DEVICES ]; then
    echo 'Set CUDA_VISIBLE_DEVICES to' $CUDA_DEVICES
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
else
    echo 'Set CUDA_VISIBLE_DEVICES to default value (0)'
    export CUDA_VISIBLE_DEVICES=0
fi

source $CONFIG_FILE

python relation_eval.py \
    --random_seed=$arg_random_seed \
    --train_file=$arg_train_file \
	--dev_file=$arg_dev_file \
	--test_file=$arg_test_file \
	--vocab_freq_file=$arg_vocab_freq_file \
	--model_dir=$arg_model_dir \
	--model_type=$arg_model_type \
    --train_label_type=$arg_train_label_type \
    --print_loss_freq=$arg_print_loss_freq \
	--max_epoch=$arg_max_epoch \
	--fix_seq_len=$arg_fix_seq_len \
	--min_word_freq=$arg_min_word_freq \
	--word_vec_size=$arg_word_vec_size \
	--pretrained_word_vectors=$arg_pretrained_word_vectors \
	--pos_max_len=$arg_pos_max_len \
	--pos_vec_size=$arg_pos_vec_size \
	--class_size=$arg_class_size \
	--hidden_size=$arg_hidden_size \
	--emb_dropout_p=$arg_emb_dropout_p \
	--lstm_dropout_p=$arg_lstm_dropout_p \
	--last_dropout_p=$arg_last_dropout_p \
	--train_batch_size=$arg_train_batch_size \
	--test_batch_size=$arg_test_batch_size \
	--optimizer_type=$arg_optimizer_type \
    --optimize_parameters=$arg_optimize_parameters \
    --mask_init_weight=$arg_mask_init_weight \
	--learning_rate=$arg_learning_rate \
	--momentum=$arg_momentum \
	--weight_decay=$arg_weight_decay \
    --lr_truncate=$arg_lr_truncate \
    --gravity=$arg_gravity \
    --truncate_freq=$arg_truncate_freq \
	--model_store_name_prefix=$arg_model_store_name_prefix \
	--model_resume_name=$arg_model_resume_name
