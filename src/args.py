# -*- coding: utf-8 -*-
# @Time    : 2/3/18 15:50
# @Author  : Shun Zheng

import argparse


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()

    default_suffix = 'spouse'
    arg_parser.add_argument('--random_seed', type=int, default=0,
                            help='Starting random seed for the program')
    arg_parser.add_argument('--use_cuda', type=bool, default=True,
                            help='Use GPU or not')
    arg_parser.add_argument('--train_file', type=str, default='./data/data_{}/train.csv'.format(default_suffix),
                            help='Training data file')
    arg_parser.add_argument('--dev_file', type=str, default='./data/data_{}/dev.csv'.format(default_suffix),
                            help='Development data file')
    arg_parser.add_argument('--test_file', type=str, default='./data/data_{}/test.csv'.format(default_suffix),
                            help='Test data file')
    arg_parser.add_argument('--vocab_freq_file', type=str, default='./data/data_{}/vocab.csv'.format(default_suffix),
                            help='Vocabulary frequency file')
    arg_parser.add_argument('--model_dir', type=str, default='./checkpoints/cpt_{}'.format(default_suffix),
                            help='Directory to save models')
    arg_parser.add_argument('--model_type', type=str, default='AttBiLSTM',
                            help='Neural network model to be used, current supported version: LSTM, AttBiLSTM')
    arg_parser.add_argument('--train_label_type', type=str, default='hard',
                            help='Train label type, hard or soft (only for binary classification')
    arg_parser.add_argument('--print_loss_freq', type=int, default=50,
                            help='Every number of mini batches to print loss score')
    arg_parser.add_argument('--max_epoch', type=int, default=50,
                            help='The maximum epoch to train')
    arg_parser.add_argument('--fix_seq_len', type=int, default=-1,
                            help='Fixed length used for model input')
    arg_parser.add_argument('--min_word_freq', type=int, default=3,
                            help='The minimum word frequency required')
    arg_parser.add_argument('--word_vec_size', type=int, default=100,
                            help='The dimension of the word vectors')
    arg_parser.add_argument('--pretrained_word_vectors', type=str, default='glove.6B.100d',
                            help='Pretrained word vectors')
    arg_parser.add_argument('--pos_max_len', type=int, default=60,
                            help='The maximum length between entity and token with unique position indicator')
    arg_parser.add_argument('--pos_vec_size', type=int, default=5,
                            help='The dimension of the position vectors')
    arg_parser.add_argument('--class_size', type=int, default=2,
                            help='The size of the target class to be predicted')
    arg_parser.add_argument('--hidden_size', type=int, default=200,
                            help='The hidden size of the LSTM cell')
    arg_parser.add_argument('--emb_dropout_p', type=float, default=0.3,
                            help='The dropout probability (p to be zeroed) for embedding output')
    arg_parser.add_argument('--lstm_dropout_p', type=float, default=0.3,
                            help='The dropout probability (p to be zeroed) for lstm output')
    arg_parser.add_argument('--last_dropout_p', type=float, default=0.5,
                            help='The dropout probability (p to be zeroed) for attentive representation')
    arg_parser.add_argument('--train_batch_size', type=int, default=50,
                            help='Batch size for training')
    arg_parser.add_argument('--test_batch_size', type=int, default=500,
                            help='Batch size for dev and test')
    arg_parser.add_argument('--optimizer_type', type=str, default='Adam',
                            choices=['SGD', 'Adagrad', 'Adadelta', 'Adam', 'TruncateSGD', 'TruncateAdam'],
                            help='Optimization method')
    arg_parser.add_argument('--optimize_parameters', type=str, default='all',
                            help='Parameters to be optimized, options: all, only_mask')
    arg_parser.add_argument('--mask_init_weight', type=str, default='random',
                            help='Mask embedding initialize choices, one or random')
    arg_parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='The learning rate of the optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.5,
                            help='The momentum of the optimizer')
    arg_parser.add_argument('--weight_decay', type=float, default=1e-5,
                            help='The weight decay of the optimizer, equivalent to L2 penalty')
    arg_parser.add_argument('--lr_truncate', type=float, default=1e-1,
                            help='The learning rate of the optimizer')
    arg_parser.add_argument('--gravity', type=float, default=1e-2,
                            help='The gravity for truncated gradient, equivalent to L1 penalty')
    arg_parser.add_argument('--truncate_freq', type=int, default=10,
                            help='The truncate frequency for truncated gradient')
    arg_parser.add_argument('--model_store_name_prefix', type=str, default='att_bi_lstm.th',
                            help='Model checkpoint file name prefix')
    arg_parser.add_argument('--model_resume_name', type=str, default='att_bi_lstm.th.best',
                            help='Model resume file name')
    arg_parser.add_argument('--model_resume_suffix', type=str, default='best',
                            help='Model resume file suffix')

    # for reinforcement erasure
    arg_parser.add_argument('--policy_train_filter_prob', type=float, default=0.5,
                            help='Filter training examples by the prediction probability')
    arg_parser.add_argument('--policy_reward_eta', type=float, default=0.5,
                            help='A hyper-parameter to weight erasure reward')
    arg_parser.add_argument('--policy_reward_gamma', type=float, default=1.0,
                            help='A hyper-parameter to penalize future rewards')
    arg_parser.add_argument('--print_reward_freq', type=int, default=1000,
                            help='Number of batches to print reward information')
    arg_parser.add_argument('--policy_max_epoch', type=int, default=5,
                            help='Max epoch for agent to train')
    arg_parser.add_argument('--policy_batch_size', type=int, default=1,
                            help='Number of examples to try in each batch')
    arg_parser.add_argument('--policy_sample_cnt', type=int, default=5,
                            help='Number of episodes to try for each example')
    arg_parser.add_argument('--policy_eps', type=float, default=0.1,
                            help='A hyper-parameter for epsilon-greedy policy')
    arg_parser.add_argument('--policy_eps_decay', type=float, default=0.9,
                            help='The decay rate for policy epsilon hyper-parameter')
    arg_parser.add_argument('--policy_store_name_prefix', type=str, default='erasure_policy',
                            help='Policy checkpoint file name prefix')
    arg_parser.add_argument('--policy_resume_name', type=str, default='erasure_policy.best',
                            help='Policy resume file name')
    arg_parser.add_argument('--policy_lr', type=float, default=1e-3,
                            help='Policy learning rate')
    arg_parser.add_argument('--policy_weight_decay', type=float, default=1e-5,
                            help='Policy weight decay (l2 regularization)')
    arg_parser.add_argument('--policy_dropout_p', type=float, default=0.5,
                            help='Policy hidden dropout probability')

    arg_parser.add_argument('--max_diag_anno', type=int, default=200,
                            help='The maximum number of labels annotated during the diagnostic process')

    tmp_args = arg_parser.parse_args(args=in_args)
    print('-'*15, 'Configuration Parameters', '-'*15)
    print(tmp_args)

    return tmp_args
