# -*- coding: utf-8 -*-
# @Time    : 27/5/18 17:27
# @Author  : Shun Zheng
# @Comment : Here 'Rule' corresponds to the notion 'Pattern' used in the paper

from __future__ import print_function

import re
import math
import pickle
import time
from collections import defaultdict
import scipy.sparse as sp_sparse
import numpy as np
import pandas as pd


policy_store_prefix_template = 'erasure_policy_model{}_fprob{}_eta{}.th'
erasure_file_name_template = 'train_erasure_decision_model{}_fprob{}_eta{}.pkl'
rule_file_name_template = 'rule_info_model{}_fprob{}_eta{}.pkl'
relation_model_prefix_template = 'rel_model{}.{}'
rule_hierarchy_file_name_template = 'rule_hierarchy_model{}.pkl'


def get_decision_rules(example_decisions, include_pad=True, include_entity=True, regex_format=False, pad_token='<pad>'):
    rule_str2ex_decs = defaultdict(lambda: [])

    for ex_dec in example_decisions:
        if ex_dec.erasure_ratio >= 1.0:
            continue

        pad_flag = False
        rule_words = []

        for word, action in zip(ex_dec.words, ex_dec.actions):
            if action == 0:  # if retain current word
                if include_pad and pad_flag:  # add pad token accordingly
                    rule_words.append(pad_token)
                pad_flag = False
                rule_words.append(word)
                continue

            if include_entity and word.startswith('ENTITY'):
                if include_pad and pad_flag:  # add pad token accordingly
                    rule_words.append(pad_token)
                pad_flag = False
                rule_words.append(word)
                continue

            pad_flag = True

        # do not include <pad> at head or tail
        # if include_pad and pad_flag:  # add pad token accordingly
        #     rule_words.append(pad_token)
        if rule_words[0] == pad_token:
            rule_words = rule_words[1:]
        if rule_words[-1] == pad_token:
            rule_words = rule_words[:-1]

        # verify if it contains meaningful words except ENTITY1, ENTITY2, <pad>
        meaning_word_flag = False
        for word in rule_words:
            if word != pad_token and not word.startswith('ENTITY'):
                meaning_word_flag = True
                break
        if not meaning_word_flag:
            continue

        rule_str = ' '.join(rule_words)
        if regex_format:
            rule_str = re.escape(rule_str).replace(re.escape(pad_token), '.*')

        rule_str2ex_decs[rule_str].append(ex_dec)

    print('Number of example decisions {}'.format(len(example_decisions)))
    print('Get {} rules in total'.format(len(rule_str2ex_decs)))

    return rule_str2ex_decs


def merge_common_rules(example_decisions, pad_token='<pad>', min_source_num=3,
                       short_pad_range=(1, 3), mid_pad_range=(4, 9)):
    assert short_pad_range[0] <= short_pad_range[1]
    assert mid_pad_range[0] <= mid_pad_range[1]
    assert short_pad_range[1] < mid_pad_range[0]

    prev_str = '[Merge Common Rules]'
    common_rule2ex_decs = defaultdict(lambda: [])

    for ex_dec in example_decisions:
        if ex_dec.erasure_ratio >= 1.0:
            continue

        pad_flag = False
        rule_words = []
        pad_index2pad_count = {}

        assert len(ex_dec.words) == len(ex_dec.actions)
        pad_cnt = 0
        for word, action in zip(ex_dec.words, ex_dec.actions):
            if action == 0 or word.startswith('ENTITY'):  # if retain current word
                if len(rule_words) > 0 and pad_flag:  # add pad token accordingly
                    rule_words.append(pad_token)
                    pad_index2pad_count[len(rule_words)-1] = pad_cnt
                rule_words.append(word)
                pad_flag = False
                pad_cnt = 0
                continue

            pad_flag = True
            pad_cnt += 1

        # do not include <pad> at head or tail
        if rule_words[0] == pad_token or rule_words[-1] == pad_token:
            print('Warning: unexpected pad token when extracting rules from example {}'.format(ex_dec.example.Id))
            continue

        # verify if it contains meaningful words except ENTITY1, ENTITY2, <pad>
        meaning_word_flag = False
        for word in rule_words:
            if word != pad_token and not word.startswith('ENTITY'):
                meaning_word_flag = True
                break
        if not meaning_word_flag:
            continue

        # no_pos_common_str = ' '.join(rule_words)
        pos_aware_common_str = ''
        for idx, token in enumerate(rule_words):
            if idx != 0:
                pos_aware_common_str += ' '

            if token != pad_token:
                pos_aware_common_str += token
            else:
                pad_cnt = pad_index2pad_count[idx]
                if short_pad_range[0] <= pad_cnt <= short_pad_range[1]:
                    pad_str = '{},{}'.format(*short_pad_range)
                elif mid_pad_range[0] <= pad_cnt <= mid_pad_range[1]:
                    pad_str = '{},{}'.format(*mid_pad_range)
                else:
                    pad_str = '{},'.format(mid_pad_range[1]+1)
                pos_aware_common_str += '{}x{}'.format(pad_token, pad_str)

        common_rule2ex_decs[pos_aware_common_str].append(ex_dec)

    print('{} Number of example decisions {}'.format(prev_str, len(example_decisions)))
    print('{} Get {} common rules in total'.format(prev_str, len(common_rule2ex_decs)))

    print('{} Filter by min_source_num < {}'.format(prev_str, min_source_num))
    rule_exdecs_list = []
    for key, val in common_rule2ex_decs.items():
        if len(val) < min_source_num:
            continue
        rule_exdecs_list.append((key, val))
    rule_exdecs_list.sort(key=lambda x: len(x[1]), reverse=True)
    print('{} Finally get {} common rules in total'.format(prev_str, len(rule_exdecs_list)))

    return rule_exdecs_list


def get_rule_quality(matched_example_row_ids, examples, target_label=1):
    matched_examples = []
    rel_cnt = 0

    for ex_row_id in matched_example_row_ids:
        ex = examples[ex_row_id]
        matched_examples.append(ex)
        if int(ex.Label) == target_label:
            rel_cnt += 1

    if len(matched_examples) > 0:
        match_acc = float(rel_cnt) / len(matched_example_row_ids)
    else:
        match_acc = -1

    return match_acc, matched_examples


def get_rule_match_stats(rule_infos, rule_name='Rules'):
    train_match_rids = []
    dev_match_rids = []
    for rule_info in rule_infos:
        train_match_rids += rule_info[3]
        dev_match_rids += rule_info[5]
    train_match_rid_set = set(train_match_rids)
    dev_match_rid_set = set(dev_match_rids)

    num_train_match = len(train_match_rids)
    num_train_match_distinct = len(train_match_rid_set)
    num_dev_match = len(dev_match_rids)
    num_dev_match_distinct = len(dev_match_rid_set)

    print('{} match {} train examples ({} distinct) and {} dev examples ({} distinct)'.format(
        rule_name, num_train_match, num_train_match_distinct, num_dev_match, num_dev_match_distinct))

    return num_train_match, num_train_match_distinct, num_dev_match, num_dev_match_distinct


def get_filtered_rules(rule_infos, filter_func, **kwargs):
    filtered_rule_infos = []
    for rule_info in rule_infos:
        if filter_func(rule_info, **kwargs):
            filtered_rule_infos.append(rule_info)
    print('Filtering condition: {}'.format(kwargs))
    print('Filtering rules from {} to {}'.format(len(rule_infos), len(filtered_rule_infos)))

    return filtered_rule_infos


def contain_meaningful_word(rule_str, pad_token='<pad>'):
    rule_words = rule_str.split(' ')
    for word in rule_words:
        if word != pad_token and not word.startswith('ENTITY'):
            return True
    return False


def filter_pos_rule_func(rule_info, min_source_num=5, min_train_match_num=5, min_dev_match_num=1, min_dev_acc=0.8):
    conds = [
        contain_meaningful_word(rule_info[0]), # contain words except ENTITY1, ENTITY2, <pad>
        len(rule_info[1]) >= min_source_num,  # enough common rule templates
        len(rule_info[3]) >= min_train_match_num,  # match enough train examples
        len(rule_info[5]) >= min_dev_match_num,  # match enough dev examples
        rule_info[6] >= min_dev_acc,  # dev match accuracy should be high enough
    ]
    if all(conds) is True:
        return True
    else:
        return False


def filter_neg_rule_func(rule_info, min_source_num=5, min_train_match_num=5, min_dev_match_num=1, max_dev_acc=0.2):
    conds = [
        contain_meaningful_word(rule_info[0]), # contain words except ENTITY1, ENTITY2, <pad>
        len(rule_info[1]) >= min_source_num,  # enough common rule templates
        len(rule_info[3]) >= min_train_match_num,  # match enough train examples
        len(rule_info[5]) >= min_dev_match_num, # match enough dev examples
        rule_info[6] <= max_dev_acc,  # dev match accuracy should be low enough
    ]
    if all(conds) is True:
        return True
    else:
        return False


# for building rule hierarchy to better filtering rules


class RuleHierarchy:
    def __init__(self, rule_str2rule_info=None, level2rule_strs=None, rule_str2level=None,
                 rule_str2children=None, rule_str2parents=None):
        self.rule_str2rule_info = rule_str2rule_info
        self.level2rule_strs = level2rule_strs
        self.rule_str2level = rule_str2level
        self.rule_str2children = rule_str2children
        self.rule_str2parents = rule_str2parents
        print('Build rule hierarchy for {} rules, {} levels, {} rules have children, {} rules have parents'.format(
            len(rule_str2rule_info), len(level2rule_strs), len(rule_str2children), len(rule_str2parents)))

    def to_dict(self):
        return {
            'rule_str2rule_info': self.rule_str2rule_info,
            'level2rule_str': self.level2rule_strs,
            'rule_str2level': self.rule_str2level,
            'rule_str2children': self.rule_str2children,
            'rule_str2parents': self.rule_str2parents,
        }


def build_rule_hierarchy(rule_infos, min_source_num=3, min_train_match_num=3, contain_ratio=0.95):
    def is_parent_child(parent_rule_info, child_rule_info):
        parent_train_eid_set = set(parent_rule_info[3])
        child_train_eid_set = set(child_rule_info[3])
        common_eid_set = child_train_eid_set.intersection(parent_train_eid_set)
        if len(child_train_eid_set) * contain_ratio < len(common_eid_set):
            return True
        else:
            return False

    def get_rule_level(rule_str, pad_token='<pad>'):
        tokens = rule_str.split(' ')
        num_level = 0
        for token in tokens:
            if not token.startswith(pad_token):
                num_level += 1
        return num_level

    rule_infos.sort(key=lambda x: (len(x[5]), len(x[3]), len(x[1])), reverse=True)

    rule_str2rule_info = {}
    level2rule_strs = {}
    rule_str2level = {}
    for rule_info in rule_infos:
        rule_str = rule_info[0]
        if len(rule_info[1]) < min_source_num:
            continue
        if len(rule_info[3]) < min_train_match_num:
            continue

        if rule_str in rule_str2rule_info:
            # use the version that have the most sources
            prev_rule_info = rule_str2rule_info[rule_str]
            if len(prev_rule_info[1]) < len(rule_info[1]):
                rule_str2rule_info[rule_str] = rule_info
            continue

        rule_str2rule_info[rule_str] = rule_info
        rule_level = get_rule_level(rule_str)

        rule_str2level[rule_str] = rule_level
        if rule_level in level2rule_strs:
            level2rule_strs[rule_level].append(rule_str)
        else:
            level2rule_strs[rule_level] = [rule_str]

    print('Filter rules from {} to {} according to match_source_num < {} and match_train_num < {}'.format(
        len(rule_infos), len(rule_str2rule_info), min_source_num, min_train_match_num))

    root_rule = 'ROOT'
    assert 1 not in level2rule_strs
    level2rule_strs[1] = [root_rule]
    rule_str2level[root_rule] = 1

    total_rule_levels = sorted(level2rule_strs.keys())
    rule_str2parents = {}
    rule_str2children = {}

    for lid, cur_level in enumerate(total_rule_levels):
        if lid == 0:
            assert cur_level == 1
            continue

        if lid == 1:
            for rule_str in level2rule_strs[cur_level]:
                rule_str2parents[rule_str] = [root_rule]
                if root_rule in rule_str2children:
                    rule_str2children[root_rule].append(rule_str)
                else:
                    rule_str2children[root_rule] = [rule_str]
            continue

        print('{} Build rule parent-child relationship for level {}'.format(time.asctime(), cur_level))

        for cur_rule_idx, rule_str in enumerate(level2rule_strs[cur_level]):
            find_parent_flag = False
            for prev_rule_str in level2rule_strs[cur_level][:cur_rule_idx]:
                prev_rule_info = rule_str2rule_info[prev_rule_str]
                rule_info = rule_str2rule_info[rule_str]
                if is_parent_child(prev_rule_info, rule_info):
                    if prev_rule_str in rule_str2children:
                        rule_str2children[prev_rule_str].append(rule_str)
                    else:
                        rule_str2children[prev_rule_str] = [rule_str]
                    if rule_str in rule_str2parents:
                        rule_str2parents[rule_str].append(prev_rule_str)
                    else:
                        rule_str2parents[rule_str] = [prev_rule_str]
                    find_parent_flag = True

            for prev_lid in range(lid - 1, 0, -1):  # do not includ 0
                if find_parent_flag:
                    break
                prev_level = total_rule_levels[prev_lid]
                for prev_rule_str in level2rule_strs[prev_level]:
                    prev_rule_info = rule_str2rule_info[prev_rule_str]
                    rule_info = rule_str2rule_info[rule_str]
                    if is_parent_child(prev_rule_info, rule_info):
                        if prev_rule_str in rule_str2children:
                            rule_str2children[prev_rule_str].append(rule_str)
                        else:
                            rule_str2children[prev_rule_str] = [rule_str]
                        if rule_str in rule_str2parents:
                            rule_str2parents[rule_str].append(prev_rule_str)
                        else:
                            rule_str2parents[rule_str] = [prev_rule_str]
                        find_parent_flag = True
            if not find_parent_flag:
                rule_str2parents[rule_str] = [root_rule]
                rule_str2children[root_rule].append(rule_str)

    rule_hierarchy = RuleHierarchy(rule_str2rule_info, level2rule_strs, rule_str2level, rule_str2children,
                                   rule_str2parents)

    return rule_hierarchy


def extract_valid_rule_infos(rule_hierarchy, is_valid_func):
    root_rule = rule_hierarchy.level2rule_strs[1][0]
    visit_rule_set = set()
    valid_rule_set = set()

    def recursive_visit(rule_str):
        visit_rule_set.add(rule_str)
        if rule_str in rule_hierarchy.rule_str2children:
            for child_rule_str in rule_hierarchy.rule_str2children[rule_str]:
                recursive_visit(child_rule_str)

    def add_valid_rule(rule_str):
        if rule_str in visit_rule_set:
            return
        rule_info = rule_hierarchy.rule_str2rule_info[rule_str]
        if is_valid_func(rule_info):
            valid_rule_set.add(rule_str)
            recursive_visit(rule_str)
        else:
            visit_rule_set.add(rule_str)
            if rule_str in rule_hierarchy.rule_str2children:
                for child_rule_str in rule_hierarchy.rule_str2children[rule_str]:
                    add_valid_rule(child_rule_str)

    if root_rule in rule_hierarchy.rule_str2children:
        for rule_str in rule_hierarchy.rule_str2children[root_rule]:
            add_valid_rule(rule_str)

    valid_rule_infos = [rule_hierarchy.rule_str2rule_info[rule_str] for rule_str in valid_rule_set]
    valid_rule_infos.sort(key=lambda x: (len(x[3]), len(x[5]), x[6]), reverse=True)

    if len(visit_rule_set) != len(rule_hierarchy.rule_str2rule_info):
        print('Warning: do not visit all rule infos when extracting valid rules')
    return valid_rule_infos


def extract_rule_to_label(rule_hierarchy, min_source_num=5, min_train_match_num=5,
                          sort_by_source_num=True):
    root_rule = rule_hierarchy.level2rule_strs[1][0]
    if root_rule in rule_hierarchy.rule_str2children:
        root_chilren = rule_hierarchy.rule_str2children[root_rule]
    else:
        root_chilren = []

    rule_infos = []
    for rule_str in root_chilren:
        rule_info = rule_hierarchy.rule_str2rule_info[rule_str]
        if len(rule_info[1]) < min_source_num:
            continue
        if len(rule_info[3]) < min_train_match_num:
            continue
        rule_infos.append(rule_info)

    if sort_by_source_num:
        rule_infos.sort(key=lambda x: (len(x[1]), len(x[3])), reverse=True)
    else:
        rule_infos.sort(key=lambda x: (len(x[3]), len(x[1])), reverse=True)

    print_rule_info_set_statstics(rule_infos, rule_name='potential rules')

    return rule_infos


def print_rule_str(rule_str, rule_hierarchy, prev_str='', min_source_num=1, min_train_match_num=1):
    rule_info = rule_hierarchy.rule_str2rule_info[rule_str]
    if len(rule_info[1]) < min_source_num:
        return
    if len(rule_info[3]) < min_train_match_num:
        return

    print(prev_str + '='*30)
    prev_str = ' '*len(prev_str)
    print(prev_str + '[LEVEL] {}'.format(rule_hierarchy.rule_str2level[rule_str]))
    print(prev_str + '[RULE] {}'.format(rule_str))
    print(prev_str + '[#TRAIN DECISION] {} [Mean Prob] {} [Example Ids] {}'.format(
        len(rule_info[1]), rule_info[2], rule_info[1][:5]))
    print(prev_str + '[#TRAIN MATCH] {} [Acc] {} [Example Ids] {}'.format(
        len(rule_info[3]), rule_info[4], rule_info[3][:5]))
    print(prev_str + '[#DEV MATCH] {} [Acc] {} [Example Ids] {}'.format(
        len(rule_info[5]), rule_info[6], rule_info[5][:5]))
    print(prev_str + '[#TEST MATCH] {} [Acc] {} [Example Ids] {}'.format(
        len(rule_info[7]), rule_info[8], rule_info[7][:5]))


def print_rule_hierarchy(rule_hierarchy, min_source_num=1, min_train_match_num=1, sort_by_source_num=True):
    def recursive_print_rule(cur_rule_str, prev_str=''):
        print_rule_str(cur_rule_str, rule_hierarchy, prev_str=prev_str,
                       min_source_num=min_source_num, min_train_match_num=min_train_match_num)
        if cur_rule_str in rule_hierarchy.rule_str2children:
            for child_rule_str in rule_hierarchy.rule_str2children[cur_rule_str]:
                recursive_print_rule(child_rule_str, prev_str=prev_str + '[Child] ')

    root_rule = rule_hierarchy.level2rule_strs[1][0]
    print('Rule hierarchy has {} rules, {} levels, {} rules have children, root rule has {} direct children'.format(
        len(rule_hierarchy.rule_str2rule_info), len(rule_hierarchy.level2rule_strs),
        len(rule_hierarchy.rule_str2children), len(rule_hierarchy.rule_str2children[root_rule])))

    root_direct_children = rule_hierarchy.rule_str2children[root_rule]
    if sort_by_source_num:
        root_direct_children.sort(key=lambda x: len(rule_hierarchy.rule_str2rule_info[x][1]), reverse=True)
    for rule_str in root_direct_children:
        recursive_print_rule(rule_str)


def print_rule_info(rule_info, prev_str=''):
    rule_str = rule_info[0]
    print('='*30, prev_str)
    print('[LEVEL] {}'.format(len(rule_str.replace('<pad>', '').split())))
    print('[RULE] {}'.format(rule_str))
    print('[#TRAIN DECISION] {} [Mean Prob] {} [Example Ids] {}'.format(
        len(rule_info[1]), rule_info[2], rule_info[1][:5]))
    print('[#TRAIN MATCH] {} [Acc] {} [Example Ids] {}'.format(
        len(rule_info[3]), rule_info[4], rule_info[3][:5]))
    print('[#DEV MATCH] {} [Acc] {} [Example Ids] {}'.format(
        len(rule_info[5]), rule_info[6], rule_info[5][:5]))
    print('[#TEST MATCH] {} [Acc] {} [Example Ids] {}'.format(
        len(rule_info[7]), rule_info[8], rule_info[7][:5]))


def print_rule_infos(rule_infos, max_print_num=500, sort_by_source_num=True):
    train_match_rids = []
    dev_match_rids = []
    for rule_info in rule_infos:
        train_match_rids += rule_info[3]
        dev_match_rids += rule_info[5]
    train_match_rids = set(train_match_rids)
    dev_match_rids = set(dev_match_rids)
    print('{} rules in total, match {} distinct train examples and {} distinct dev examples'.format(
        len(rule_infos), len(train_match_rids), len(dev_match_rids)))

    if sort_by_source_num:
        rule_infos.sort(key=lambda x: len(x[1]), reverse=True)
    else:
        rule_infos.sort(key=lambda x: len(x[3]), reverse=True)

    for idx, rule_info in enumerate(rule_infos):
        if idx >= max_print_num:
            break
        print_rule_info(rule_info, prev_str=str(idx))


def print_rule_info_set_statstics(rule_infos, rule_name='rules'):
    train_match_eids = []
    dev_match_eids = []
    for rule_info in rule_infos:
        train_match_eids += rule_info[3]
        dev_match_eids += rule_info[5]
    train_match_eid_set = set(train_match_eids)
    dev_match_eid_set = set(dev_match_eids)

    num_train_match = len(train_match_eids)
    num_train_match_distinct = len(train_match_eid_set)
    num_dev_match = len(dev_match_eids)
    num_dev_match_distinct = len(dev_match_eid_set)

    print('[{} statistics]:'.format(rule_name).upper())
    print('{} {} match {} train examples ({} distinct) and {} dev examples ({} distinct)'.format(
        len(rule_infos), rule_name, num_train_match, num_train_match_distinct, num_dev_match, num_dev_match_distinct))


# for creating training data from rules

def truncate_prob(raw_prob, eps=1e-5):
    if raw_prob >= 1 - eps:
        prob = 1 - eps
    elif raw_prob <= eps:
        prob = eps
    else:
        prob = raw_prob

    return prob


def estimate_rule_parameter(rule_infos, target_label, num_train_example, avg_rule_acc=0.8):
    # the avg_rule_acc is used when we want to manually add some rules,
    # but their approximate quality measures are not good enough.
    eps = 1e-5
    gen_paras = []
    for rule_info in rule_infos:
        beta_coef = float(len(rule_info[3])) / num_train_example

        if target_label == 1:
            alpha_coef = rule_info[6]

            if alpha_coef > 1 + eps or alpha_coef <= 0.5:
                print('Dynamically change positive rule "{}" accuracy from {} to {}'.format(
                    rule_info[0], alpha_coef, avg_rule_acc))
                alpha_coef = avg_rule_acc
        elif target_label == -1:
            alpha_coef = 1 - rule_info[6]

            if alpha_coef > 1 + eps or alpha_coef <= 0.5:
                print('Dynamically change negative rule "{}" accuracy from {} to {}'.format(
                    rule_info[0], alpha_coef, avg_rule_acc))
                alpha_coef = avg_rule_acc
        else:
            raise ValueError('target label must be in [-1, 1], while get {}'.format(target_label))
        assert 0 <= alpha_coef <= 1
        assert 0 <= beta_coef <= 1
        alpha_coef = truncate_prob(alpha_coef)
        beta_coef = truncate_prob(beta_coef)

        gen_paras.append((alpha_coef, beta_coef))

    return gen_paras


def build_label_matrix(raw_train_examples, pos_rule_infos, neg_rule_infos, use_example_id=True):
    example_id2row_id = {}
    for row_id, ex in enumerate(raw_train_examples):
        example_id2row_id[int(ex.Id)] = row_id

    tmp_rows = []
    tmp_cols = []
    tmp_data = []

    col_id = 0
    for row_id, example in enumerate(raw_train_examples):
        tmp_rows.append(row_id)
        tmp_cols.append(col_id)
        if example.Label == '1':
            label = 1
        else:
            label = -1
        tmp_data.append(label)

    col_pre_num = 1
    for rule_id, rule_info in enumerate(pos_rule_infos):
        col_id = col_pre_num + rule_id
        for row_id in rule_info[3]:  # train matched example row indexes
            if use_example_id:
                row_id = example_id2row_id[row_id]
            tmp_rows.append(row_id)
            tmp_cols.append(col_id)
            tmp_data.append(1)

    col_pre_num += len(pos_rule_infos)
    for rule_id, rule_info in enumerate(neg_rule_infos):
        col_id = col_pre_num + rule_id
        for row_id in rule_info[3]:
            if use_example_id:
                row_id = example_id2row_id[row_id]
            tmp_rows.append(row_id)
            tmp_cols.append(col_id)
            tmp_data.append(-1)

    num_rows = len(raw_train_examples)
    num_cols = col_pre_num + len(neg_rule_infos)

    rule_label_mat = sp_sparse.coo_matrix((tmp_data, (tmp_rows, tmp_cols)), shape=(num_rows, num_cols))
    rule_label_csr_mat = rule_label_mat.tocsr()
    print('Build rule augmented label matrix (shape={}, nnzs={})'.format(
        rule_label_csr_mat.shape, rule_label_csr_mat.nnz))

    return rule_label_csr_mat


class RuleGenerativeModel(object):
    def __init__(self, rule_paras, rule_names=None):
        self.rule_paras = rule_paras
        self.alpha_coefs = [para[0] for para in rule_paras]
        self.beta_coefs = [para[1] for para in rule_paras]
        self.rule_names = rule_names

        print('Build Label Fusion Model')
        for rule, para in zip(self.rule_names, self.rule_paras):
            try:
                print('{:50}\t{}'.format(rule, para))
            except:
                try:
                    # print('{:50}\t{}'.format(unicode(rule).encode('utf-8'), para))
                    # for python 3
                    print('{:50}\t{}'.format(rule, para))
                except:
                    # print('{:50}\t{} has invalid unicode characters'.format(
                    # unicode(rule).encode('ascii', 'ignore'), para))
                    # for python 3
                    print('{:50}\t{} has invalid unicode characters'.format(rule, para))
                    continue

    def weighted_inference(self, rule_label_mat):
        num_rows, num_cols = rule_label_mat.shape
        assert num_cols == len(self.rule_paras)
        rule_label_mat = rule_label_mat.tocsr()

        pos_marginals = []
        for row_id in range(num_rows):
            row_label_mat = rule_label_mat[row_id, :]
            rule_ids = row_label_mat.indices
            rule_labels = row_label_mat.data

            joint_pos_log_prob = 0.0
            joint_neg_log_prob = 0.0
            for rule_id, rule_label in zip(rule_ids, rule_labels):
                rule_pos_prob = self.alpha_coefs[rule_id] * int(rule_label == 1) + \
                                (1 - self.alpha_coefs[rule_id]) * int(rule_label == -1)
                rule_neg_prob = self.alpha_coefs[rule_id] * int(rule_label == -1) + \
                                (1 - self.alpha_coefs[rule_id]) * int(rule_label == 1)
                joint_pos_log_prob += math.log(truncate_prob(rule_pos_prob))
                joint_neg_log_prob += math.log(truncate_prob(rule_neg_prob))
            pos_over_neg_prob = math.exp(joint_pos_log_prob - joint_neg_log_prob)
            pos_cond_prob = pos_over_neg_prob / (1.0 + pos_over_neg_prob)
            pos_marginals.append(pos_cond_prob)

        return pos_marginals

    @staticmethod
    def majority_vote(rule_label_mat, follow_rule_flag=True):
        num_rows, num_cols = rule_label_mat.shape
        rule_label_mat = rule_label_mat.tocsr()

        vote_labels = []
        for row_id in range(num_rows):
            row_label_mat = rule_label_mat[row_id, :]
            rule_labels = row_label_mat.data

            label_sum = np.sum(rule_labels)
            if label_sum == 0:
                if follow_rule_flag:
                    label_sum = np.sum(rule_labels[1:])
                else:
                    label_sum = rule_labels[0]

            if label_sum > 0:
                vote_labels.append(1)
            elif label_sum < 0:
                vote_labels.append(0)
            else:
                raise ValueError('Vote label sum should not be equal to 0')

        return vote_labels


def transform_labeled_data(raw_csv_path, new_csv_path, new_labels):
    print('Transform labels from {} into {}'.format(raw_csv_path, new_csv_path))
    df = pd.read_csv(raw_csv_path, header=None)
    df.columns = ['Id', 'Text', 'Pos1', 'Pos2', 'Label']
    df['Label'] = new_labels
    df.to_csv(new_csv_path, header=False, index=False)


def build_label_from_rule(rel_env, rule_pkl_file_path, raw_data_file_path,
                          pos_filter_func=filter_pos_rule_func,
                          neg_filter_func=filter_neg_rule_func,
                          ndp_data_file_path=None, vote_data_file_path=None):
    reward_eta = rule_pkl_file_path.split('_eta')[-1].rstrip('.pkl')
    print('-' * 5, 'rule file {}, reward eta {}'.format(rule_pkl_file_path, reward_eta))
    with open(rule_pkl_file_path, 'rb') as fin:
        rule_infos = pickle.load(fin)

    pos_rule_infos = get_filtered_rules(rule_infos, pos_filter_func)
    get_rule_match_stats(pos_rule_infos, rule_name='Positive rules')

    neg_rule_infos = get_filtered_rules(rule_infos, neg_filter_func)
    get_rule_match_stats(neg_rule_infos, rule_name='Negative Rules')

    rule_label_csr_mat = build_label_matrix(rel_env.raw_train_examples, pos_rule_infos, neg_rule_infos)

    num_train_example = len(rel_env.raw_train_examples)
    pos_rule_paras = estimate_rule_parameter(pos_rule_infos, 1, num_train_example)
    neg_rule_paras = estimate_rule_parameter(neg_rule_infos, -1, num_train_example)
    total_rule_paras = [(0.8, 1.0)] + pos_rule_paras + neg_rule_paras
    total_rule_names = ['DistantSupervision'] + ['[POS] ' + rule_info[0] for rule_info in pos_rule_infos] + [
        '[NEG] ' + rule_info[0] for rule_info in neg_rule_infos]

    rule_gen_model = RuleGenerativeModel(total_rule_paras, rule_names=total_rule_names)

    train_marginals = rule_gen_model.weighted_inference(rule_label_csr_mat)
    train_vote_labels = rule_gen_model.majority_vote(rule_label_csr_mat)

    if ndp_data_file_path is not None:
        transform_labeled_data(raw_data_file_path, ndp_data_file_path, train_marginals)

    if vote_data_file_path is not None:
        transform_labeled_data(raw_data_file_path, vote_data_file_path, train_vote_labels)

    return train_marginals, train_vote_labels


def create_diagnosed_train_data(raw_train_examples, pos_rule_infos, neg_rule_infos,
                                raw_data_path, new_data_path=None, ds_param=(0.8, 1.0)):
    rule_label_csr_mat = build_label_matrix(raw_train_examples, pos_rule_infos, neg_rule_infos, use_example_id=True)

    # estimate parameters for data programming
    num_train_examples = len(raw_train_examples)
    pos_rule_paras = estimate_rule_parameter(pos_rule_infos, 1, num_train_examples)
    neg_rule_paras = estimate_rule_parameter(neg_rule_infos, -1, num_train_examples)

    # build generative model to infer label
    total_rule_paras = [ds_param] + pos_rule_paras + neg_rule_paras
    total_rule_names = ['DistantSupervision'] + ['[POS] ' + rule_info[0] for rule_info in pos_rule_infos] + [
        '[NEG] ' + rule_info[0] for rule_info in neg_rule_infos]
    rule_gen_model = RuleGenerativeModel(total_rule_paras, rule_names=total_rule_names)
    new_labels = rule_gen_model.weighted_inference(rule_label_csr_mat)

    if new_data_path is not None:
        transform_labeled_data(raw_data_path, new_data_path, new_labels)

    return new_labels


