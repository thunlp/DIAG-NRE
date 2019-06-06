#!/bin/bash

CONFIG_FILE=$1
if [ $CONFIG_FILE ]; then
    echo 'Using config file', $CONFIG_FILE
else
    echo 'Please give a config file as the first argument'
    echo "Usage: $0 config_file [gpu_id]"
    exit 1
fi

source $CONFIG_FILE

CUDA_DEVICES=$2
if [ $CUDA_DEVICES ]; then
    echo 'Set CUDA_VISIBLE_DEVICES to' $CUDA_DEVICES
else
    CUDA_DEVICES=0
    echo 'Set CUDA_VISIBLE_DEVICES to default value (0)'
fi

NEW_LOG_NAME="$LOG_NAME.fprob$arg_policy_train_filter_prob.eta$arg_policy_reward_eta"
EXEC_COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m agent_train \
    --random_seed=$arg_random_seed \
	--train_file=$arg_train_file \
	--dev_file=$arg_dev_file \
	--test_file=$arg_test_file \
	--vocab_freq_file=$arg_vocab_freq_file \
	--model_dir=$arg_model_dir \
	--model_store_name_prefix=$arg_model_store_name_prefix \
	--model_resume_name=$arg_model_resume_name \
    --model_resume_suffix=$arg_model_resume_suffix \
    --policy_train_filter_prob=$arg_policy_train_filter_prob \
    --policy_reward_eta=$arg_policy_reward_eta \
    --policy_reward_gamma=$arg_policy_reward_gamma \
    --print_reward_freq=$arg_print_reward_freq \
    --policy_max_epoch=$arg_policy_max_epoch \
    --policy_batch_size=$arg_policy_batch_size \
    --policy_sample_cnt=$arg_policy_sample_cnt \
    --policy_eps=$arg_policy_eps \
    --policy_eps_decay=$arg_policy_eps_decay \
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
    &>> $DIR_PREFIX/logs/$NEW_LOG_NAME"

export PATH="$PYTHON_BIN_DIR:$PATH"

echo "Execute $EXEC_COMMAND"
eval "$EXEC_COMMAND"

