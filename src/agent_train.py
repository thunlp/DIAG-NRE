# -*- coding: utf-8 -*-
# @Time    : 21/5/18 09:17
# @Author  : Shun Zheng

from __future__ import print_function

import os
import pickle
import time

from rule_helpers import policy_store_prefix_template, erasure_file_name_template, rule_file_name_template
from agent_task import RelationEnv, EraseAgent, aggregate_rule_info
from args import parse_args
from utils import apply_gpu_memory


def deep_rule_extraction(rel_args, train_agent_flag=True):
    print('{} [DEEP RULE EXTRACTION] starts'.format(time.asctime()))

    # build environment
    rel_env = RelationEnv(rel_args)

    # adhoc modification
    # rel_env.config['policy_train_filter_prob'] = 1e4

    model_type = rel_env.model_type
    reward_eta = rel_env.config['policy_reward_eta']
    train_filter_prob = rel_env.config['policy_train_filter_prob']

    model_suffix = rel_env.config['model_resume_suffix']
    model_str = '{}-{}'.format(model_type, model_suffix)

    erasure_saved_file = erasure_file_name_template.format(model_str, train_filter_prob, reward_eta)
    rule_saved_file = rule_file_name_template.format(model_str, train_filter_prob, reward_eta)

    erasure_file_path = os.path.join(rel_env.config['model_dir'], erasure_saved_file)
    rule_file_path = os.path.join(rel_env.config['model_dir'], rule_saved_file)

    if train_agent_flag or not os.path.exists(erasure_file_path) or not os.path.exists(rule_file_path):
        # 0. Filter training examples for erasure agent
        rel_env.filter_environment_train_set(pred_filter_prob=train_filter_prob)
        # 1. Agent training with different reward_eta
        policy_store_prefix = policy_store_prefix_template.format(model_str, train_filter_prob, reward_eta)
        agent = EraseAgent(rel_env.config, rel_env.state_size, rel_env.device, resume_flag=False)
        agent.batch_train(rel_env, policy_store_prefix=policy_store_prefix)
        # 2. Agent evaluation to get erasure decisions on training set
        train_example_decisions = agent.batch_eval(rel_env, saved_pickle_file=erasure_saved_file)
    else:
        # directly resume train decisions from previous dumped file
        with open(erasure_file_path, 'rb') as fin:
            train_example_decisions = pickle.load(fin)
        print('Resume train erasure decisions from {} and continue'.format(erasure_file_path))

    # 3. Aggregate rule information and filter rules
    rule_infos = aggregate_rule_info(rel_env, train_example_decisions,
                                     num_label_proc=10,
                                     saved_pickle_file=rule_saved_file)

    print('{} [DEEP RULE EXTRACTION] ends'.format(time.asctime()))

    return rule_infos


if __name__ == '__main__':
    # parse input arguments
    in_args = parse_args()
    # quickly apply gpu memory
    apply_gpu_memory(gpu_id=0)

    # deep_rule_extraction(in_args, train_agent_flag=False)
    deep_rule_extraction(in_args, train_agent_flag=True)

