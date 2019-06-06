# -*- coding: utf-8 -*-
# @Time    : 25/7/18 22:47
# @Author  : Shun Zheng

from itertools import product

from batch_common import batch_do_task, TOTAL_REL_DIRS, \
    AGENT_TRAIN_COMMAND_TEMPLATE, AGENT_CONFIG_BASE, AGENT_CONFIG_NAME_TEMPLATE
from rule_helpers import relation_model_prefix_template


if __name__ == '__main__':
    total_task_rel_args = []

    train_type = 'train_ds'
    model_type = 'AttBiLSTM'
    for msuf in ['1', '2', '3']:
        for eta in [0.05, 0.1, 0.5, 1.0, 1.5]:
            for fprob in [1e4]:
                task_args = [{
                    'TRAIN_TYPE': train_type,
                    'arg_policy_reward_eta': eta,
                    'arg_policy_train_filter_prob': fprob,
                    'arg_model_type': model_type,
                    'arg_model_store_name_prefix': relation_model_prefix_template.format(model_type, train_type),
                    'arg_model_resume_name': relation_model_prefix_template.format(model_type, train_type) + '.{}'.format(msuf),
                    'arg_model_resume_suffix': msuf,
                }]

                total_task_rel_args += list(product(TOTAL_REL_DIRS, task_args))

    batch_do_task(total_task_rel_args,
                  AGENT_CONFIG_BASE, AGENT_CONFIG_NAME_TEMPLATE, AGENT_TRAIN_COMMAND_TEMPLATE,
                  max_gpu_mem_usage=0.01)
