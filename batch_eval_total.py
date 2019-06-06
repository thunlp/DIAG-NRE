# -*- coding: utf-8 -*-
# @Time    : 29/7/18 16:35
# @Author  : Shun Zheng

from itertools import product

from batch_common import batch_do_task, TOTAL_REL_DIRS, \
    TOTAL_EVAL_COMMAND_TEMPLATE, AGENT_CONFIG_BASE, AGENT_CONFIG_NAME_TEMPLATE


if __name__ == '__main__':
    total_task_rel_args = []

    train_type = 'train_ds'
    dev_type = 'dev_diag'
    test_type = 'test_human'
    model_type = 'AttBiLSTM'

    task_args = [{
        'TRAIN_TYPE': train_type,
        'DEV_TYPE': dev_type,
        'TEST_TYPE': test_type,
        'arg_model_type': model_type,
    }]
    total_task_rel_args += list(product(TOTAL_REL_DIRS, task_args))

    batch_do_task(total_task_rel_args,
                  AGENT_CONFIG_BASE, AGENT_CONFIG_NAME_TEMPLATE, TOTAL_EVAL_COMMAND_TEMPLATE,
                  max_gpu_mem_usage=0.3)
