# -*- coding: utf-8 -*-
# @Time    : 25/7/18 22:47
# @Author  : Shun Zheng

from batch_common import batch_do_task, TOTAL_REL_DIRS, \
    REL_TRAIN_COMMAND_TEMPLATE, REL_CONFIG_BASE, REL_CONFIG_NAME_TEMPLATE
from rule_helpers import relation_model_prefix_template


if __name__ == '__main__':
    total_task_rel_args = []

    model_type = 'AttBiLSTM'
    label_type = 'soft'
    train_type = 'train_ds'
    max_epoch = 3

    # After generating all kinds of training labels,
    # use the following code with multiple random seeds to produce final results
    for random_seed in range(5):
        model_str = model_type + '_seed{}'.format(random_seed)
        for train_type in ['train_ds', 'train_diag_mda200']:
            # mda corresponds to the maximum diagnostic annotation
            for rel_dir in TOTAL_REL_DIRS:
                task_arg = {
                    'arg_random_seed': random_seed,
                    'TRAIN_TYPE': train_type,
                    'arg_model_store_name_prefix': relation_model_prefix_template.format(model_str, train_type),
                    'arg_model_type': model_type,
                    'arg_max_epoch': max_epoch,
                    'arg_train_label_type': label_type,
                }
                total_task_rel_args.append((rel_dir, task_arg))

    batch_do_task(total_task_rel_args,
                  REL_CONFIG_BASE, REL_CONFIG_NAME_TEMPLATE, REL_TRAIN_COMMAND_TEMPLATE,
                  max_gpu_mem_usage=0.05)

