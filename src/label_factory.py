# -*- coding: utf-8 -*-
# @Time    : 23/5/18 15:59
# @Author  : Shun Zheng
# @Comment : Part of the codes was inspired by Snorkel(https://github.com/HazyResearch/snorkel)

import re
import time
# from Queue import Empty
# for python3
from queue import Empty
from multiprocessing import JoinableQueue, Process, Manager
import scipy.sparse as sp_sparse
import itertools


QUEUE_TIMEOUT = 3
PROCESS_EXIT_SIGNAL = 'Labeler Exit'


def create_label_func(rule_str, output_label, regex_format=False, include_pad=True, pad_range=True, pad_token='<pad>'):
    if regex_format:
        # note that sometimes regular expression can be very slow
        regex_obj = re.compile(rule_str, flags=re.UNICODE)

        def label_func(example):
            sent_text = ' '.join(example.Text)
            if regex_obj.match(sent_text):
                return output_label
            else:
                return 0
    else:
        if include_pad:
            if pad_range:
                tokens = rule_str.split(' ')
                max_pad_num = 500  # no sentence is longer than 500
                padded_rule_words = []
                rule_pad_index2pad_range = {}
                rule_word_set = set()
                for idx, token in enumerate(tokens):
                    if token.startswith(pad_token):
                        pad_str, pad_range_str = token.split('x')
                        assert pad_str == pad_token
                        pad_min_str, pad_max_str = pad_range_str.split(',')
                        if len(pad_max_str) > 0:
                            rule_pad_index2pad_range[idx] = (int(pad_min_str), int(pad_max_str))
                        else:
                            rule_pad_index2pad_range[idx] = (int(pad_min_str), max_pad_num)
                        padded_rule_words.append(pad_token)
                    else:
                        rule_word_set.add(token)
                        padded_rule_words.append(token)

                def label_func(example):
                    ex_words = example.Text
                    # build example template and pad count statistics
                    ex_template = []
                    ex_pad_index2pad_count = {}
                    pad_cnt = 0
                    pad_flag = False
                    for ex_word in ex_words:
                        if ex_word in rule_word_set:
                            if pad_flag:
                                ex_template.append(pad_token)
                                ex_pad_index2pad_count[len(ex_template)-1] = pad_cnt
                            ex_template.append(ex_word)
                            pad_flag = False
                            pad_cnt = 0
                        else:
                            pad_flag = True
                            pad_cnt += 1

                    if len(ex_template) < len(padded_rule_words):
                        return 0

                    for ex_idx in range(len(ex_template)-len(padded_rule_words)+1):
                        match_flag = True
                        for j in range(len(padded_rule_words)):
                            if ex_template[ex_idx+j] != padded_rule_words[j]:
                                match_flag = False
                                break
                            if padded_rule_words[j] == pad_token:
                                ex_pad_cnt = ex_pad_index2pad_count[ex_idx+j]
                                rule_pad_range = rule_pad_index2pad_range[j]
                                if ex_pad_cnt < rule_pad_range[0] or ex_pad_cnt > rule_pad_range[1]:
                                    match_flag = False
                                    break
                        if match_flag:
                            return output_label

                    return 0
            else:
                pad_rule_words = rule_str.split(' ')
                rule_word_set = set([word for word in pad_rule_words if word != pad_token])
                # assert pad_rule_words[0] != pad_token
                # assert pad_rule_words[-1] != pad_token
                assert len(rule_word_set) >= 1

                def label_func(example):
                    ex_words = example.Text
                    ex_template = []
                    pad_flag = False
                    for ex_word in ex_words:
                        if ex_word in rule_word_set:
                            if pad_flag:
                                ex_template.append(pad_token)
                            pad_flag = False
                            ex_template.append(ex_word)
                        else:
                            pad_flag = True
                    # if pad_flag:
                    #     ex_template.append(pad_token)

                    if len(ex_template) < len(pad_rule_words):
                        return 0

                    for ex_idx in range(len(ex_template)-len(pad_rule_words)+1):
                        match_flag = True
                        for j in range(len(pad_rule_words)):
                            if ex_template[ex_idx+j] != pad_rule_words[j]:
                                match_flag = False
                                break
                        if match_flag:
                            return output_label
                    return 0
        else:
            rule_words = [word for word in rule_str.split(' ') if word != pad_token]
            assert len(rule_words) >= 1

            def label_func(example):
                ex_words = example.Text
                rule_match_idx = 0
                for ex_word in ex_words:
                    if rule_match_idx < len(rule_words):
                        if ex_word == rule_words[rule_match_idx]:
                            rule_match_idx += 1
                    else:
                        break

                if rule_match_idx >= len(rule_words):
                    return output_label
                else:
                    return 0

    return label_func


class BatchLabeler(Process):
    def __init__(self, worker_id, label_func_ids, label_funcs, examples, in_queue, out_queue):
        super(BatchLabeler, self).__init__()
        self.daemon = True

        self.worker_id = worker_id
        self.label_func_ids = label_func_ids
        self.label_funcs = label_funcs
        self.examples = examples
        self.in_queue = in_queue
        self.out_queue = out_queue

    def annotate(self, example_mb_range, func_mb_range):
        cur_label_func_ids = self.label_func_ids[func_mb_range[0]:func_mb_range[1]]
        cur_label_funcs = self.label_funcs[func_mb_range[0]:func_mb_range[1]]
        cur_examples = self.examples[example_mb_range[0]:example_mb_range[1]]

        annotation_bag = []
        for example in cur_examples:
            example_id = example.Id
            annotations = []
            for label_func_id, label_func in zip(cur_label_func_ids, cur_label_funcs):
                if label_func is None:
                    continue
                anno_label = label_func(example)
                if anno_label != 0:
                    annotations.append((label_func_id, anno_label))
            annotation_bag.append((example_id, annotations))

        return annotation_bag

    def run(self):
        print('{} BatchLabeler worker {} runs with {} label functions and {} examples'.format(
            time.asctime(), self.worker_id, len(self.label_funcs), len(self.examples)))

        task_cnt = 0
        print_run_time_thresh = 60 * 60  # 1 hour
        start_time = time.time()
        pivot_time = start_time

        while True:
            try:
                in_task = self.in_queue.get(True, QUEUE_TIMEOUT)
                if in_task == PROCESS_EXIT_SIGNAL:
                    break

                example_mb_range, func_mb_range = in_task
                # print('{} BatchLabeler worker {} get task {}x{}'.format(
                #     time.asctime(), self.worker_id, example_mb_range, func_mb_range))
                annotation_bag = self.annotate(example_mb_range, func_mb_range)
                self.out_queue.put(annotation_bag, True, QUEUE_TIMEOUT)
                self.in_queue.task_done()

                task_cnt += 1

                cur_time = time.time()
                run_time = cur_time - pivot_time
                total_run_time = cur_time - start_time
                if run_time > print_run_time_thresh:
                    pivot_time = cur_time
                    print('{} Worker {} has completed {} tasks, total time cost {:.3f}min'.format(
                        time.asctime(), self.worker_id, task_cnt, total_run_time/60))
            except Empty:
                time.sleep(0.1)

        total_run_time = time.time() - start_time
        print('{0} BatchLabeler worker {1} exits (complete {2} tasks, total time cost {3:.3f}min) successfully'.format(
            time.asctime(), self.worker_id, task_cnt, total_run_time/60))


def _get_num_proc_and_mini_batch_ranges(batch_size, num_proc):
    if batch_size < 20 or batch_size < num_proc:
        new_num_proc = 1
        mini_batch_ranges = [(0, batch_size)]
    else:
        new_num_proc = num_proc
        mb_size_proc = batch_size // num_proc
        mini_batch_ranges = []
        for idx in range(num_proc):
            mb_s = idx * mb_size_proc
            mb_e = (idx + 1) * mb_size_proc
            # avoid missing the tail
            if idx == num_proc - 1:
                mb_e = batch_size
            mini_batch_ranges.append((mb_s, mb_e))

    return new_num_proc, mini_batch_ranges


def _get_num_task_and_mini_batch_ranges(batch_size, num_task):
    if batch_size < 20 or batch_size < num_task:
        new_num_task = 1
        mini_batch_ranges = [(0, batch_size)]
    else:
        new_num_task = num_task
        mb_size_task = batch_size // num_task
        mini_batch_ranges = []
        for idx in range(num_task):
            mb_s = idx * mb_size_task
            mb_e = (idx + 1) * mb_size_task
            # avoid missing the tail
            if idx == num_task - 1:
                mb_e = batch_size
            mini_batch_ranges.append((mb_s, mb_e))

    return new_num_task, mini_batch_ranges


def _terminate_label_procs(label_procs, hard_flag=False):
    for proc in label_procs:
        if not hard_flag:
            proc.join()
        proc.terminate()


class LabelFactory(object):
    def __init__(self, label_funcs, num_label_proc=8):
        self.label_funcs = label_funcs
        self.label_func_ids = range(len(label_funcs))
        self.num_label_proc = num_label_proc

        print('Create label factory with {} label functions'.format(
            len(self.label_funcs)))

    def reset_label_funcs(self, new_label_funcs):
        print('Reset labeling functions, from {} to {}'.format(len(self.label_funcs), len(new_label_funcs)))
        self.label_funcs = new_label_funcs

    def batch_annotate(self, examples, num_example_per_task=100, num_func_per_task=100, num_label_proc=None, matrix_format='coo_matrix'):
        if len(self.label_funcs) == 0:
            print('No labeling functions, do nothing')
            return
        if len(examples) == 0:
            print('No examples to be annotated, do nothing')
            return
        if num_label_proc is None:
            num_label_proc = self.num_label_proc

        num_rows, num_cols = len(examples), len(self.label_funcs)
        row_id2example_id = [example.Id for example in examples]
        example_id2row_id = dict(zip(row_id2example_id, range(num_rows)))
        example_label_results = []

        print('{} Label factory batch annotate starts'.format(time.asctime()))

        batch_task_queue = JoinableQueue()
        batch_annotation_queue = JoinableQueue()

        batch_label_procs = []
        num_example_tasks = max(num_rows // num_example_per_task, 1)
        num_example_tasks, example_mb_ranges = _get_num_task_and_mini_batch_ranges(num_rows, num_example_tasks)
        num_func_tasks = max(num_cols // num_func_per_task, 1)
        num_func_tasks, func_mb_ranges = _get_num_task_and_mini_batch_ranges(num_cols, num_func_tasks)

        for example_mb_range, func_mb_range in itertools.product(example_mb_ranges, func_mb_ranges):
            batch_task_queue.put((example_mb_range, func_mb_range))
        for _ in range(num_label_proc):
            batch_task_queue.put(PROCESS_EXIT_SIGNAL)

        num_total_tasks = num_example_tasks * num_func_tasks
        print('{} Label factory pushes {} tasks ({}x{}) into the task queue'.format(
            time.asctime(), num_total_tasks, num_example_tasks, num_func_tasks))

        for pid in range(num_label_proc):
            proc = BatchLabeler(worker_id=pid,
                                label_func_ids=self.label_func_ids,
                                label_funcs=self.label_funcs,
                                examples=examples,
                                in_queue=batch_task_queue,
                                out_queue=batch_annotation_queue)
            batch_label_procs.append(proc)

        print('{} Batch annotation processes start, {} examples, {} functions, {} processes'.format(
            time.asctime(), num_rows, num_cols, num_label_proc))

        for proc in batch_label_procs:
            proc.start()

        tmp_idx = 0
        queue_print_freq = max(num_total_tasks // 10, 100)
        # wait for all batch labeling processes to finish
        while any([proc.is_alive() for proc in batch_label_procs]):
            while True:
                try:
                    if tmp_idx % queue_print_freq == 0:
                        num_remain_task = batch_task_queue.qsize()
                        ratio_remain_task = 100 * num_remain_task // num_total_tasks
                        print('{} Out queue gets {} times, collect {} results, {} tasks ({}%) to be done'.format(
                            time.asctime(), tmp_idx, len(example_label_results), num_remain_task, ratio_remain_task))
                    tmp_idx += 1

                    annotation_bag = batch_annotation_queue.get(True, QUEUE_TIMEOUT)
                    example_label_results += annotation_bag
                    batch_annotation_queue.task_done()
                except Empty:
                    time.sleep(0.1)
                    if batch_annotation_queue.qsize() == 0:
                        break

        print('{} Out queue gets {} results in total'.format(time.asctime(), len(example_label_results)))
        _terminate_label_procs(batch_label_procs)
        print('{} Batch annotation processes exit'.format(time.asctime()))

        if len(example_label_results) != num_rows * num_func_tasks:
            print('Warning: do not obtain all annotation results, task queue {}, annotation queue {}'.format(
                batch_task_queue.qsize(), batch_annotation_queue.qsize()))

        # extract example label results to construct a sparse matrix
        tmp_rows = []
        tmp_cols = []
        tmp_data = []
        for example_id, label_res in example_label_results:
            for lf_id, lf_label in label_res:
                assert lf_label in [-1, 1]
                row_id = example_id2row_id[example_id]

                tmp_rows.append(row_id)
                tmp_cols.append(lf_id)
                tmp_data.append(lf_label)

        label_mat = sp_sparse.coo_matrix((tmp_data, (tmp_rows, tmp_cols)), shape=(num_rows, num_cols))
        if matrix_format == 'coo_matrix':
            return label_mat
        elif matrix_format == 'csr_matrix':
            return label_mat.tocsr()
        elif matrix_format == 'csc_matrix':
            return label_mat.tocsc()
        elif matrix_format == 'lil_matrix':
            return label_mat.tolil()
        else:
            print('Unknown matrix format {}, return coo_matrix format instead'.format(matrix_format))
            return label_mat
