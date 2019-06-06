# -*- coding: utf-8 -*-
# @Time    : 12/6/18 10:22
# @Author  : Shun Zheng

from __future__ import print_function

import os
import pickle
import shutil
from args import parse_args
from rule_helpers import relation_model_prefix_template
from agent_task import RelationTask
from utils import resume_and_evaluate, apply_gpu_memory


def total_eval_relation_models(rel_task):
    print('-'*15, 'Recover relation models and do total evaluation', '-'*15)
    model_type = rel_task.model_type

    rel_model_prefix = relation_model_prefix_template.split('.')[0].format(model_type)
    model_name2test_pr = {}
    model_name2test_pred_info = {}
    eval_results = []
    for file_name in os.listdir(rel_task.config['model_dir']):
        conds = [
            file_name.startswith(rel_model_prefix),
            file_name.endswith('.best'),
            ]
        if all(conds) is True:
            model_cpt_path = os.path.join(rel_task.config['model_dir'], file_name)

            dev_eval_results = resume_and_evaluate(rel_task, model_cpt_path, rel_task.dev_iter)
            dev_pr_auc= dev_eval_results[3]
            dev_avg_prec = dev_eval_results[4]
            dev_f1_score = dev_eval_results[5]

            test_eval_results = resume_and_evaluate(rel_task, model_cpt_path, rel_task.test_iter)
            test_pr_auc = test_eval_results[3]
            test_avg_prec = test_eval_results[4]
            test_f1_score = test_eval_results[5]
            test_prec = test_eval_results[6]
            test_recall = test_eval_results[7]
            model_name2test_pr[file_name] = (test_eval_results[1], test_eval_results[2], test_prec, test_recall)
            model_name2test_pred_info[file_name] = test_eval_results[0]

            eval_results.append((file_name,
                                 dev_avg_prec, test_avg_prec,
                                 test_prec, test_recall, test_f1_score))

    eval_results.sort(key=lambda x: x[-1], reverse=True)
    print('='*50)
    print('Total evaluation results:')
    print('{:50}\tDev AP\tTest AP\tTest P\tTest R\tTest F1'.format('Model File Name'))
    for res in eval_results:
        print('{:50}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
            res[0].lstrip('rel_model'), res[1], res[2], res[3], res[4], res[5]))

    def dump_into_pkl_file(obj, out_file_path, obj_name=''):
        with open(out_file_path, 'wb') as fout:
            pickle.dump(obj, fout)
        print('-'*5, 'Dump {} into {}'.format(obj_name, out_file_path))

    test_pred_info_file = 'eval_model{}_test_pred_info.pkl'.format(model_type)
    out_path = os.path.join(rel_task.config['model_dir'], test_pred_info_file)
    dump_into_pkl_file(model_name2test_pred_info, out_path, obj_name='test prediction info')

    test_pr_curve_file = 'eval_model{}_test_pr_curve.pkl'.format(model_type)
    out_path = os.path.join(rel_task.config['model_dir'], test_pr_curve_file)
    dump_into_pkl_file(model_name2test_pr, out_path, obj_name='test precision-recall curves')


def total_select_best_model(rel_task):
    """Reselect the best epoch suitable for the test data"""
    print('-'*15, 'Batch reselect the best model', '-'*15)
    model_type = rel_task.model_type
    model_dir = rel_task.config['model_dir']

    rel_model_prefix = relation_model_prefix_template.split('.')[0].format(model_type)
    file_base2files = {}
    for file_name in os.listdir(model_dir):
        if file_name.startswith(rel_model_prefix):
            last_term = file_name.split('.')[-1]
            file_base = file_name.rstrip(last_term)
            if file_base in file_base2files:
                file_base2files[file_base].append(file_name)
            else:
                file_base2files[file_base] = [file_name]

    for file_base, files in file_base2files.items():
        print('-'*30)
        print('File base {}'.format(file_base))
        cur_best_f1 = 0.0
        best_model_path = os.path.join(model_dir, file_base + 'best')
        for file_name in files:
            model_cpt_path = os.path.join(model_dir, file_name)
            if model_cpt_path == best_model_path:
                continue
            test_eval_results = resume_and_evaluate(rel_task, model_cpt_path, rel_task.test_iter)
            test_f1_score = test_eval_results[5]
            if test_f1_score > cur_best_f1:
                cur_best_f1 = test_f1_score
                print('Current best F1 {}, copy {} to {}'.format(cur_best_f1, model_cpt_path, best_model_path))
                shutil.copy(model_cpt_path, best_model_path)


if __name__ == '__main__':
    in_args = parse_args()
    # quickly apply gpu memory
    apply_gpu_memory(gpu_id=0)

    # create relation task
    rel_task = RelationTask(in_args, resume_flag=False, dump_flag=True)

    total_eval_relation_models(rel_task)
