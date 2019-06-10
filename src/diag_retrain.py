# -*- coding: utf-8 -*-
# @Time    : 3/8/18 16:23
# @Author  : Shun Zheng

from __future__ import print_function

import os
import pickle
import time

from agent_task import RelationTask, RelationEnv, EraseAgent, aggregate_rule_info, update_rule_info
from args import parse_args
from rule_helpers import erasure_file_name_template, rule_file_name_template, relation_model_prefix_template, \
    policy_store_prefix_template, build_rule_hierarchy, extract_valid_rule_infos, print_rule_info_set_statstics, \
    create_diagnosed_train_data, rule_hierarchy_file_name_template
from utils import retrain_and_evaluate, apply_gpu_memory


def batch_eval_agent_erasure(rel_args):
    print('-'*15, 'Batch resume agent and do erasure evaluation', '-'*15)
    # build environment
    rel_env = RelationEnv(rel_args)
    model_type = rel_env.model_type
    agent = EraseAgent(rel_env.config, rel_env.state_size, rel_env.device, resume_flag=False)
    policy_file_prefix = policy_store_prefix_template.split('_fprob')[0].format(model_type)

    model_dir = rel_env.config['model_dir']
    for file_name in os.listdir(model_dir):
        if file_name.startswith(policy_file_prefix) and file_name.endswith('.th.best'):
            best_policy_path = os.path.join(model_dir, file_name)
            agent.resume_policy_from(best_policy_path)

            fprob_eta_term = file_name.replace('.th.best', '')
            fprob_term, eta_str = fprob_eta_term.split('_eta')
            fprob_str = fprob_term.split('_fprob')[-1]
            fprob, eta = float(fprob_str), float(eta_str)

            rel_env.set_reward_eta(eta)
            rel_env.filter_environment_train_set(pred_filter_prob=fprob)

            erasure_dump_file = erasure_file_name_template.format(model_type, fprob, eta)
            agent.batch_eval(rel_env, saved_pickle_file=erasure_dump_file)


def batch_aggregate_rule_info(rel_args):
    print('-'*15, 'Batch resume decisions and aggregate rule information', '-'*15)
    # build environment
    rel_env = RelationEnv(rel_args)
    model_type = rel_env.model_type
    erasure_file_prefix = erasure_file_name_template.split('_fprob')[0].format(model_type)

    model_dir = rel_env.config['model_dir']
    for file_name in os.listdir(model_dir):
        if file_name.startswith(erasure_file_prefix):  # only consider train decision files
            # read train decisions from the train decision file
            erasure_decision_fp = os.path.join(model_dir, file_name)
            print('-'*5, 'Recover train decision from file {}'.format(erasure_decision_fp))
            with open(erasure_decision_fp, 'rb') as fin:
                train_example_decisions = pickle.load(fin)

            fprob_eta_term = file_name.replace('.pkl', '')
            fprob_term, eta_str = fprob_eta_term.split('_eta')
            fprob_str = fprob_term.split('_fprob')[-1]
            fprob, eta = float(fprob_str), float(eta_str)

            rule_saved_file = rule_file_name_template.format(model_type, fprob, eta)
            aggregate_rule_info(rel_env, train_example_decisions,
                                num_label_proc=10,
                                min_source_num=3,
                                saved_pickle_file=rule_saved_file)
    print('-'*15, 'Batch aggregate rule information ends', '-'*15)


def batch_update_rule_info(rel_args):
    print('-'*15, 'Batch update rule statistics information', '-'*15)
    # build environment
    rel_env = RelationEnv(rel_args)

    # use all kinds of rule files
    # rule_file_prefix = rule_file_name_template.split('{}')[0]

    # use certain rule files (fix model and fprob)
    # train_fprob = rel_env.config['policy_train_filter_prob']
    # rule_file_prefix = rule_file_name_template.split('_eta')[0].format(rel_env.model_type, train_fprob)

    # use certain rule files (fix model)
    rule_file_prefix = rule_file_name_template.split('_fprob')[0].format(rel_env.model_type)

    max_diag_anno = rel_env.config['max_diag_anno']
    raw_diag_anno = len(rel_env.raw_dev_examples)
    if max_diag_anno < raw_diag_anno:
        print('Change dev annotated examples from {} to {}'.format(raw_diag_anno, max_diag_anno))
        rel_env.raw_dev_examples = rel_env.raw_dev_examples[:max_diag_anno]

    model_dir = rel_env.config['model_dir']
    for file_name in os.listdir(model_dir):
        if file_name.startswith(rule_file_prefix):  # only consider train decision files
            # read train decisions from the train decision file
            rule_file_path = os.path.join(model_dir, file_name)
            print('-'*5, 'Recover rule infos from file {}'.format(rule_file_path))
            with open(rule_file_path, 'rb') as fin:
                prev_rule_infos = pickle.load(fin)

            rule_saved_file = file_name
            update_rule_info(rel_env, prev_rule_infos,
                             num_label_proc=10,
                             saved_pickle_file=rule_saved_file,
                             train_flag=False, dev_flag=True, test_flag=False)

    print('-'*15, 'Batch update rule statistics information ends', '-'*15)


def build_rule_hierarchy_from(rule_pkl_dir, rule_file_filter_str='rule_info_modelAttBiLSTM',
                              resume_file_name=None, dump_file_name=None,
                              min_source_num=3, min_train_match_num=3):
    if resume_file_name is not None:
        rule_hierarchy_path = os.path.join(rule_pkl_dir, resume_file_name)
        with open(rule_hierarchy_path, 'rb') as fin:
            rule_hierarchy = pickle.load(fin)
        print('Resume rule hierarchy from {}'.format(rule_hierarchy_path))
        return rule_hierarchy

    # load rule infos
    rule_files = os.listdir(rule_pkl_dir)
    total_rule_infos = []
    for file_name in rule_files:
        # if file_name.startswith('rule_info_fprob0.5_eta1.5'):
        # if rule_file_filter_str in file_name:
        if file_name.startswith(rule_file_filter_str):
            file_path = os.path.join(rule_pkl_dir, file_name)
            with open(file_path, 'rb') as fin:
                rule_infos = pickle.load(fin)
            print('{} Load rule infos from pickle file {}'.format(time.asctime(), file_path))
            total_rule_infos += rule_infos

    # build rule hierarchy
    rule_hierarchy = build_rule_hierarchy(total_rule_infos,
                                          min_source_num=min_source_num,
                                          min_train_match_num=min_train_match_num)

    if dump_file_name is not None:
        dump_fp = os.path.join(rule_pkl_dir, dump_file_name)
        with open(dump_fp, 'wb') as fout:
            pickle.dump(rule_hierarchy, fout)
        print('Dump rule hierarchy into {}'.format(dump_fp))

    return rule_hierarchy


def rule_refinement(rule_pkl_dir=None, rule_file_filter_str='rule_info_modelAttBiLSTM', rule_hierarchy=None,
                    dump_hierarchy_file=None, pos_filter_cond=None, neg_filter_cond=None):
    assert rule_pkl_dir is not None or rule_hierarchy is not None
    print('[RULE REFINEMENT] starts')

    if rule_pkl_dir is not None and rule_hierarchy is None:
        rule_hierarchy = build_rule_hierarchy_from(rule_pkl_dir,
                                                   rule_file_filter_str=rule_file_filter_str,
                                                   dump_file_name=dump_hierarchy_file,
                                                   min_source_num=3,
                                                   min_train_match_num=3)

    # filter positive rules
    if pos_filter_cond is None:
        pos_filter_cond = {
            'min_source_num': 5,
            'min_train_match_num': 5,
            'min_dev_match_num': 3,
            'min_dev_acc': 0.8,
        }
    print('Positive filtering condition {}'.format(pos_filter_cond))

    def is_pos_rule(rule_info):
        conds = [
            len(rule_info[1]) >= pos_filter_cond['min_source_num'],
            len(rule_info[3]) >= pos_filter_cond['min_train_match_num'],
            len(rule_info[5]) >= pos_filter_cond['min_dev_match_num'],
            rule_info[6] >= pos_filter_cond['min_dev_acc']
        ]
        if all(conds) is True:
            return True
        else:
            return False

    pos_rule_infos = extract_valid_rule_infos(rule_hierarchy, is_pos_rule)
    print_rule_info_set_statstics(pos_rule_infos, rule_name='positive rules')

    # filter negative rules
    if neg_filter_cond is None:
        neg_filter_cond = {
            'min_source_num': 5,
            'min_train_match_num': 5,
            'min_dev_match_num': 3,
            'max_dev_acc': 0.1,
        }
    print('Negative filtering condition {}'.format(neg_filter_cond))

    def is_neg_rule(rule_info):
        conds = [
            len(rule_info[1]) >= neg_filter_cond['min_source_num'],
            len(rule_info[3]) >= neg_filter_cond['min_train_match_num'],
            len(rule_info[5]) >= neg_filter_cond['min_dev_match_num'],
            rule_info[6] <= neg_filter_cond['max_dev_acc']
        ]
        if all(conds) is True:
            return True
        else:
            return False

    neg_rule_infos = extract_valid_rule_infos(rule_hierarchy, is_neg_rule)
    print_rule_info_set_statstics(neg_rule_infos, rule_name='negative rules')

    print('[RULE REFINEMENT] ends')
    return pos_rule_infos, neg_rule_infos


def rule_diagnosis_and_retrain(rel_args, pos_rule_infos=None, neg_rule_infos=None,
                               resume_manual_flag=False, adaptive_ds_flag=False,
                               pos_filter_cond=None, neg_filter_cond=None):
    def dump_rule_info(rule_infos, dump_dir, dump_file_name, dump_obj_name='rule infos'):
        dump_path = os.path.join(dump_dir, dump_file_name)
        with open(dump_path, 'wb') as fout:
            pickle.dump(rule_infos, fout)
        print('Dump {} into {}'.format(dump_obj_name, dump_path))
    print('{} [RULE DIAGNOSIS AND RETRAIN] starts'.format(time.asctime()))
    train_type_template = 'train_diag_{}'
    pos_rule_file_template = 'pos_rule_info_model{}_{}.pkl'
    neg_rule_file_template = 'neg_rule_info_model{}_{}.pkl'
    train_filter_prob = rel_args.policy_train_filter_prob
    model_type = rel_args.model_type
    rule_pkl_dir = rel_args.model_dir

    if resume_manual_flag:
        tmp_suffix = 'manual'

        pos_rule_path = os.path.join(rule_pkl_dir, pos_rule_file_template.format(model_type, tmp_suffix))
        with open(pos_rule_path, 'rb') as fin:
            pos_rule_infos = pickle.load(fin)
        print('Load position rule infos from {}'.format(pos_rule_path))

        neg_rule_path = os.path.join(rule_pkl_dir, neg_rule_file_template.format(model_type, tmp_suffix))
        with open(neg_rule_path, 'rb') as fin:
            neg_rule_infos = pickle.load(fin)
        print('Load negative rule infos from {}'.format(neg_rule_path))

    if pos_rule_infos is None or neg_rule_infos is None:
        if pos_filter_cond is None:
            pos_filter_cond = {
                'min_source_num': 5,
                'min_train_match_num': 5,
                'min_dev_match_num': 10,
                'min_dev_acc': 0.8,
            }

        if neg_filter_cond is None:
            neg_filter_cond = {
                'min_source_num': 5,
                'min_train_match_num': 5,
                'min_dev_match_num': 10,
                'max_dev_acc': 0.1,
            }

        # build rule hierarchy
        # rule_file_filter_str = rule_file_name_template.split('_eta')[0].format(model_type, train_filter_prob)
        rule_file_filter_str = rule_file_name_template.split('_fprob')[0].format(model_type)
        dump_hierarchy_file = rule_hierarchy_file_name_template.format(model_type)
        rule_hierarchy = build_rule_hierarchy_from(rule_pkl_dir,
                                                   rule_file_filter_str=rule_file_filter_str,
                                                   dump_file_name=dump_hierarchy_file)

        # rule refinement
        pos_rule_infos, neg_rule_infos = rule_refinement(rule_hierarchy=rule_hierarchy,
                                                         pos_filter_cond=pos_filter_cond,
                                                         neg_filter_cond=neg_filter_cond)
    else:
        print('Use existing positive and negative rules')
        print_rule_info_set_statstics(pos_rule_infos, rule_name='positive rules')
        print_rule_info_set_statstics(neg_rule_infos, rule_name='negative rules')

    if len(pos_rule_infos) == 0 and len(neg_rule_infos) == 0:
        print('Do not get valuable rules, return directly')
        return None

    # create relation task
    rel_task = RelationTask(rel_args, resume_flag=False, dump_flag=True)

    # prepare output file names
    if pos_filter_cond is None or neg_filter_cond is None:
        tmp_suffix = 'manual'
        train_type = train_type_template.format(tmp_suffix)
    else:
        max_diag_anno = rel_args.max_diag_anno
        tmp_suffix = 'mda{}'.format(max_diag_anno)  # the maximum number of diagnostic annotations used
        train_type = train_type_template.format(tmp_suffix)
        pos_rule_file = pos_rule_file_template.format(model_type, tmp_suffix)
        neg_rule_file = neg_rule_file_template.format(model_type, tmp_suffix)
        dump_rule_info(pos_rule_infos, rule_pkl_dir, pos_rule_file, dump_obj_name='positive rule infos')
        dump_rule_info(neg_rule_infos, rule_pkl_dir, neg_rule_file, dump_obj_name='negative rule infos')

    raw_train_path = rel_task.config['train_file']
    data_dir = os.path.dirname(raw_train_path)
    # estimate ds parameters
    if adaptive_ds_flag:
        # this option is deprecated because the selected data are highly biased,
        # which is improper for estimations of DS parameters.
        ds_acc = estimate_distant_supervision_accuracy(data_dir, rel_task.train_set.examples,
                                                       dev_anno_file='rule_anno_dev.csv')
        train_type += '_adapt'
    else:
        # default choise is to assign DS with a fixed prior parameter
        ds_acc = 0.8

    new_train_path = os.path.join(data_dir, '{}.csv'.format(train_type))

    # create diagnosed training data
    create_diagnosed_train_data(rel_task.train_set.examples, pos_rule_infos, neg_rule_infos, raw_train_path,
                                new_data_path=new_train_path, ds_param=(ds_acc, 1.0))

    # # retrain relation model
    # new_model_prefix = relation_model_prefix_template.format(rel_task.model_type, train_type)
    # retrain_and_evaluate(rel_task, new_train_path, new_model_prefix, rel_task.dev_iter)

    print('{} [RULE DIAGNOSIS AND RETRAIN] ends'.format(time.asctime()))


def estimate_distant_supervision_accuracy(data_dir_path, raw_train_examples, dev_anno_file='rule_anno_dev.csv'):
    eid2ds_label = {}
    for ex in raw_train_examples:
        eid2ds_label[int(ex.Id)] = int(ex.Label)

    import pandas as pd
    dev_anno_path = os.path.join(data_dir_path, dev_anno_file)
    dev_anno_df = pd.read_csv(dev_anno_path, header=None)
    print('Load {}'.format(dev_anno_path))
    dev_anno_df.columns = ['Id', 'Text', 'Pos1', 'Pos2', 'Label']
    dev_anno_instances = dev_anno_df.to_dict(orient='records')
    acc = 0.0
    for ins in dev_anno_instances:
        eid = int(ins['Id'])
        if eid2ds_label[eid] == int(ins['Label']):
            acc += 1
    acc /= len(dev_anno_instances)
    print('Estimate distant supervision accuracy as {}'.format(acc))

    return acc


if __name__ == '__main__':
    # parse input arguments
    in_args = parse_args()
    # quickly apply gpu memory
    apply_gpu_memory(gpu_id=0)

    # batch_eval_agent_erasure(in_args)
    # batch_aggregate_rule_info(in_args)
    batch_update_rule_info(in_args)

    rule_diagnosis_and_retrain(in_args, resume_manual_flag=False, adaptive_ds_flag=False)


