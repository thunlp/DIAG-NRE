# -*- coding: utf-8 -*-
# @Time    : 3/3/18 16:42
# @Author  : Shun Zheng

from __future__ import print_function

from args import parse_args
from relation_task import RelationTask

if __name__ == '__main__':
    rel_args = parse_args()
    rel_task = RelationTask(rel_args,
                            train_flag=False,
                            dev_flag=True,
                            test_flag=True,
                            resume_flag=True,
                            dump_flag=False)
    rel_task.dump_prediction_info(rel_task.dev_iter, file_name='dev_prediction.csv')
    rel_task.dump_prediction_info(rel_task.test_iter, file_name='test_prediction.csv')
