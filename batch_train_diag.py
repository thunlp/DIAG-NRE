# -*- coding: utf-8 -*-
# @Time    : 3/8/18 16:31
# @Author  : Shun Zheng

from itertools import product

from batch_common import batch_do_task, TOTAL_REL_DIRS, \
    DIAGNOSIS_CONFIG_BASE, DIAGNOSIS_CONFIG_NAME_TEMPALTE, DIAGNOSIS_COMMAND_TEMPLATE


if __name__ == '__main__':
    total_task_rel_args = []

    train_type = 'train_ds'
    train_fprob = 1e4
    task_args = [{
        'TRAIN_TYPE': train_type,
        'arg_policy_train_filter_prob': train_fprob,
    }]
    total_task_rel_args += list(product(TOTAL_REL_DIRS, task_args))

    batch_do_task(total_task_rel_args,
                  DIAGNOSIS_CONFIG_BASE, DIAGNOSIS_CONFIG_NAME_TEMPALTE, DIAGNOSIS_COMMAND_TEMPLATE,
                  max_gpu_mem_usage=0.05)

