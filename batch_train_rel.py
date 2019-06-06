# -*- coding: utf-8 -*-
# @Time    : 25/7/18 22:47
# @Author  : Shun Zheng

from itertools import product
import os

from batch_common import batch_do_task, TOTAL_REL_DIRS, \
    REL_TRAIN_COMMAND_TEMPLATE, REL_CONFIG_BASE, REL_CONFIG_NAME_TEMPLATE
from rule_helpers import relation_model_prefix_template


if __name__ == '__main__':
    total_task_rel_args = []

    model_type = 'AttBiLSTM'
    label_type = 'soft'
    train_type = 'train_ds'
    max_epoch = 3

    for rel_dir in TOTAL_REL_DIRS:
        task_arg = {
            'TRAIN_TYPE': train_type,
            'arg_model_store_name_prefix': relation_model_prefix_template.format(model_type, train_type),
            'arg_model_type': model_type,
            'arg_max_epoch': max_epoch,
            'arg_train_label_type': label_type,
        }

        total_task_rel_args.append((rel_dir, task_arg))

    if not os.path.exists('.vector_cache'):
        # just start one job to prepare the cache for Glove embeddings
        batch_do_task(total_task_rel_args[:1],
                      REL_CONFIG_BASE, REL_CONFIG_NAME_TEMPLATE, REL_TRAIN_COMMAND_TEMPLATE,
                      max_gpu_mem_usage=0.05)
        # Later jobs can directly utilize the existing cache
        batch_do_task(total_task_rel_args[1:],
                      REL_CONFIG_BASE, REL_CONFIG_NAME_TEMPLATE, REL_TRAIN_COMMAND_TEMPLATE,
                      max_gpu_mem_usage=0.05)
    else:
        batch_do_task(total_task_rel_args,
                      REL_CONFIG_BASE, REL_CONFIG_NAME_TEMPLATE, REL_TRAIN_COMMAND_TEMPLATE,
                      max_gpu_mem_usage=0.05)

