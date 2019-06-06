# -*- coding: utf-8 -*-
# @Time    : 3/3/18 16:42
# @Author  : Shun Zheng

from __future__ import print_function

from args import parse_args
from relation_task import RelationTask
from utils import apply_gpu_memory

if __name__ == '__main__':
    rel_args = parse_args()
    # quickly bind a gpu
    apply_gpu_memory(gpu_id=0)

    rel_task = RelationTask(rel_args, train_flag=True, dev_flag=True, test_flag=True,
                            resume_flag=True, dump_flag=True)

    rel_task.train()
