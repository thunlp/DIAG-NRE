# -*- coding: utf-8 -*-
# @Time    : 5/3/18 10:24
# @Author  : Shun Zheng

from __future__ import print_function

import csv
import sys
import os
import shutil
import time
import random
from collections import Counter, OrderedDict
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtext.vocab import pretrained_aliases, Vectors
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, auc


# a temporary hack for using chinese word vectors in the financial domain
class MyVec(Vectors):
    def __init__(self, name='wind_vec.50d.txt', **kwargs):
        super(MyVec, self).__init__(name, **kwargs)


def build_field_vocab_from_dict(field, vocab_freq_dict, **kwargs):
    print('Build vocabulary from dict')
    vocab_counter = Counter(vocab_freq_dict)
    # taken from torchtext.data.Field.build_vocab()
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(vocab_counter, specials=specials, **kwargs)
    print('Total vocabulary size:', len(field.vocab.freqs), 'effective size:', len(field.vocab))


def build_field_vocab_from_file(field, vocab_freq_file, **kwargs):
    print('Build Field vocabulary from existing vocabulary files', vocab_freq_file)
    # read vocabulary frequency file
    with open(vocab_freq_file, 'r') as fin:
        csv_reader = csv.reader(fin)
        vocab_freq = []
        for row in csv_reader:
            word, freq = row
            # word = word.decode('utf-8')
            # for python3
            freq = int(freq)
            vocab_freq.append((word, freq))
    vocab_counter = Counter(dict(vocab_freq))

    # taken from torchtext.data.Field.build_vocab()
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(vocab_counter, specials=specials, **kwargs)
    print('Total vocabulary size:', len(field.vocab.freqs), 'effective size:', len(field.vocab))


def build_field_vocab_from_dataset(field, data_set, vocab_freq_file=None, **kwargs):
    # build field vocabulary from data_set
    print('Build Field vocabulary from dataset')
    field.build_vocab(data_set, **kwargs)
    print('Total vocabulary size:', len(field.vocab.freqs), 'effective size:', len(field.vocab))

    if isinstance(vocab_freq_file, str):
        print('Dump vocabulary frequencies into', vocab_freq_file)
        # dump vocabulary frequency
        sorted_vocab_freq = sorted(field.vocab.freqs.items(), key=lambda x: x[1])
        with open(vocab_freq_file, 'w') as fout:
            csv_writer = csv.writer(fout)
            for row in sorted_vocab_freq:
                word, freq = row
                # word = word.encode('utf-8')
                # for python3
                csv_writer.writerow([word, freq])


def build_field_vocabulary(field,
                           from_vocab=True,
                           vocab_freq_file=None,
                           vocab_freq_dict=None,
                           data_set=None,
                           **kwargs):
    """
    Build field vocabulary with three options:
    1. from vocabulary frequency file
    2. from vocabulary frequency dict
    3. by counting tokens in dataset and dump into vocab_freq_file accordingly

    Args:
        field: torchtext.data.Field object
        from_vocab: flag of whether to recover from the vocabulary file directly
        vocab_freq_file: the absolute path of the vocabulary file
        vocab_freq_dict: the vocabulary frequency dictionary
        data_set: torchtext.data.Dataset object
        **kwargs: key word arguments to be parsed to torchtext.Vocab class
    """
    if 'vectors' in kwargs and isinstance(kwargs['vectors'], str) and kwargs['vectors'] not in pretrained_aliases:
        print('Read from self-pretrained vectors', kwargs['vectors'])
        kwargs['vectors'] = MyVec(name=kwargs['vectors'])

    if from_vocab and isinstance(vocab_freq_file, str) and os.path.exists(vocab_freq_file):
        build_field_vocab_from_file(field, vocab_freq_file, **kwargs)
    elif from_vocab and isinstance(vocab_freq_dict, dict):
        build_field_vocab_from_dict(field, vocab_freq_dict, **kwargs)
    elif data_set is not None:
        build_field_vocab_from_dataset(field, data_set, vocab_freq_file, **kwargs)
    else:
        raise Exception('Build field vocabulary failed, please check input arguments!')


def random_init_certain_vector(vocab, token='<unk>', mean=0, std=0.5):
    """
    Randomly initialize certain vector of the vocabulary object

    Args:
        vocab: the object of torchtext.vocab.Vocab class
        token: token string
        mean: mean of the normal distribution
        std: std of the normal distribution
    """
    idx = vocab.stoi[token]
    nn.init.normal_(vocab.vectors[idx], mean=mean, std=std)


def save_checkpoint(state_dict, is_best, file_path_prefix, file_name_suffix=''):
    file_path = file_path_prefix + file_name_suffix
    torch.save(state_dict, file_path)
    if is_best:
        shutil.copyfile(file_path, file_path_prefix + '.best')


def resume_checkpoint(net, model_file_path,
                      strict=False, resume_key='model', print_keys=('epoch', 'dev_f1', 'dev_avg_prec')):
    if os.path.exists(model_file_path):
        resume_dict = torch.load(model_file_path)
        print('Resume from previous model checkpoint {}'.format(model_file_path))
        for key in print_keys:
            if key in resume_dict:
                print('{}: {}'.format(key, resume_dict[key]))
        net.load_state_dict(resume_dict[resume_key], strict=strict)
        print('Resume successfully')
        return True
    else:
        print(Warning('Warning: model resume failed because', model_file_path, 'not found'))
        return False


def id_to_word(word_ids, itos):
    words = []
    for wid in word_ids:
        words.append(itos[wid])

    return words


def show_word_score_heatmap(score_tensor, x_ticks, y_ticks, figsize=(3, 8)):
    # to make colorbar a proper size w.r.t the image
    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.1)
        return fig.colorbar(mappable, cax=cax)

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    img = ax.matshow(score_tensor.numpy())

    plt.xticks(range(score_tensor.size(1)), x_ticks, fontsize=14)
    plt.yticks(range(score_tensor.size(0)), y_ticks, fontsize=14)

    colorbar(img)

    ax.set_aspect('auto')
    plt.show()


def show_word_scores_heatmap(score_tensor_tup, x_ticks, y_ticks, nrows=1, ncols=1, titles=None, figsize=(8, 8), fontsize=14):
    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.1)
        return fig.colorbar(mappable, cax=cax)
    if not isinstance(score_tensor_tup, tuple):
        score_tensor_tup = (score_tensor_tup, )

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for idx, ax in enumerate(axs):
        score_tensor = score_tensor_tup[idx]
        img = ax.matshow(score_tensor.numpy())
        plt.sca(ax)
        plt.xticks(range(score_tensor.size(1)), x_ticks, fontsize=fontsize)
        plt.yticks(range(score_tensor.size(0)), y_ticks, fontsize=fontsize)
        if titles is not None:
            plt.title(titles[idx], fontsize=fontsize + 2)
        colorbar(img)

    for ax in axs:
        ax.set_aspect('auto')
    plt.tight_layout(h_pad=1)

    plt.show()


def build_salient_phrase_candidates(sample_decomp_info, num_classes=2, score_threshold=1.1):
    # logic to filter unimportant phrases
    def is_salient_phrase(phrase_scores, num_classes, score_threshold):
        for cid in range(num_classes):
            x = phrase_scores[:, cid] > score_threshold
            if x.sum() == len(x):
                # all values for cid are greater than the threshold
                return True
        return False

    phrase_candidate_dict = defaultdict(lambda: {'sample_ids': [],
                                                 'decompose_scores': [],
                                                 'count': None,
                                                 'average_score': None,
                                                 'phrase_score': None,
                                                 'class_id': None})

    # iterate through all samples to construct phrase candidates
    for idx, decomp_sample in enumerate(sample_decomp_info):
        sample_id = decomp_sample[0]
        word_ids = decomp_sample[1]
        decomp_scores = np.array(decomp_sample[2])

        for idx in range(len(word_ids)):
            ngram_max = len(word_ids) - idx
            for ngram_len in range(1, ngram_max + 1):
                sid = idx
                eid = idx + ngram_len
                tmp_phr_scores = decomp_scores[sid:eid, :]

                if is_salient_phrase(tmp_phr_scores, num_classes, score_threshold):
                    tmp_phr_ids = tuple(word_ids[sid:eid])
                    tmp_phr_total_score = np.prod(tmp_phr_scores, axis=0, keepdims=True)
                    # record salient phrase candidate
                    phrase_candidate_dict[tmp_phr_ids]['sample_ids'].append(sample_id)
                    phrase_candidate_dict[tmp_phr_ids]['decompose_scores'].append(tmp_phr_total_score)
                else:
                    # because later ngrams cannot be salient phrases
                    break

    # calculate average scores and associated information for each phrase
    for key in phrase_candidate_dict:
        phr_dict = phrase_candidate_dict[key]
        # get expected decomposition score
        avg_score = np.mean(np.concatenate(phr_dict['decompose_scores'], axis=0), axis=0)
        # normalize
        avg_score = avg_score / avg_score.sum()
        max_score = np.max(avg_score)
        class_id = np.argmax(avg_score)
        phr_dict['count'] = len(phr_dict['sample_ids'])
        phr_dict['average_score'] = avg_score
        phr_dict['phrase_score'] = max_score
        phr_dict['class_id'] = class_id

    return phrase_candidate_dict


def get_salient_phrases(phrase_candidate_dict, word_id2str=None, num_classes=2, min_count=0):
    salient_phrases = [[] for _ in range(num_classes)]
    for phrase_ids_key in phrase_candidate_dict:
        phr_dict = phrase_candidate_dict[phrase_ids_key]
        if phr_dict['count'] < min_count:
            continue
        if word_id2str is None:
            words = None
        else:
            words = id_to_word(phrase_ids_key, word_id2str)
        # Note: this is not a deep copy, it just creates a new dict() object,
        # but values in the dictionary refer previous memory spaces.
        new_phr_dict = {'phrase_ids': phrase_ids_key, 'words': words}
        for key in ['sample_ids', 'phrase_score']:
            new_phr_dict[key] = phr_dict[key]  # note: this is a shallow copy

        cid = phr_dict['class_id']
        salient_phrases[cid].append(new_phr_dict)

    for phr_dicts in salient_phrases:
        phr_dicts.sort(key=lambda x: x['phrase_score'], reverse=True)

    return salient_phrases


def get_confusion_matrix(raw_ids, pred_probs, true_labels, threshold=0.5):
    tps = []
    fps = []
    fns = []
    tns = []
    for idx, rid in enumerate(raw_ids):
        p = pred_probs[idx]
        if p >= threshold:
            if true_labels[idx] == 1:
                tps.append((rid, p))
            elif true_labels[idx] == 0:
                fps.append((rid, p))
            else:
                raise ValueError('Value for the label must be 1 or 0')
        else:
            if true_labels[idx] == 1:
                fns.append((rid, p))
            elif true_labels[idx] == 0:
                tns.append((rid, p))
            else:
                raise ValueError('Value for the label must be 1 or 0')

    return tps, fps, fns, tns


def resume_and_evaluate(rel_task, cpt_file_path, rel_dataset_iter):
    print('{} Resume and evaluate'.format(time.asctime(), cpt_file_path))
    if cpt_file_path is not None:
        rel_task.resume_model_from(cpt_file_path, strict=True)

    eids, pred_probs, true_labels = rel_task.get_prediction_probs_info(rel_dataset_iter)
    example_ids = eids.tolist()
    pred_probs = pred_probs[:, 1].numpy()
    true_labels = true_labels.numpy()
    pred_labels = (pred_probs > 0.5).astype(int)
    pred_info = {
        'example_ids': example_ids,
        'true_labels': true_labels,
        'pred_probs': pred_probs,
        'pred_labels': pred_labels
    }

    precs, recalls, threshes = precision_recall_curve(true_labels, pred_probs)
    pr_auc = auc(recalls, precs)
    avg_prec = average_precision_score(true_labels, pred_probs)
    dec_prec, dec_recall, dec_f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')

    print('[Evaluate Results]: prec {:.3f}, recall {:.3f}, f1 {:.3f}, avg prec {:.3f}, pr auc {:.3f}'.format(
        dec_prec, dec_recall, dec_f1_score, avg_prec, pr_auc))

    return pred_info, precs, recalls, pr_auc, avg_prec, dec_f1_score, dec_prec, dec_recall


def retrain_and_evaluate(rel_task, new_train_file, model_store_prefix, rel_dataset_iter,
                         print_loss_freq=1000):
    print('{} Re-train relation task with {}'.format(time.asctime(), new_train_file))

    rel_task.config['train_file'] = new_train_file
    rel_task.config['model_store_name_prefix'] = model_store_prefix
    rel_task.init_train_set()  # read new training data
    rel_task.init_neural_network()  # initialize neural network, optimizer, loss, dev state

    rel_task.train(print_loss_freq=print_loss_freq)

    best_cpt_path = os.path.join(rel_task.config['model_dir'],
                                 '{}.best'.format(model_store_prefix))
    dev_eval_results = resume_and_evaluate(rel_task, best_cpt_path, rel_dataset_iter)
    dev_f1_score = dev_eval_results[5]

    print('{} Re-train procedure completes, dev f1 score is {}'.format(time.asctime(), dev_f1_score))

    return dev_f1_score


def plot_multi_pr_curves(plot_tuples, plot_title='Precision Recall Curves',
                         figsize=(12, 8), xlim=(0, 1), ylim=(0, 1),
                         basic_font_size=14):
    plt.figure(figsize=figsize)

    for eval_infos, line_name, line_color in plot_tuples:
        precs = eval_infos[0]
        recalls = eval_infos[1]
        avg_prec = eval_infos[3]
        f1_score = eval_infos[6]
        plt.step(recalls, precs,
                 label=line_name + ' (AUC {0:.3f}, F1 {1:.3f})'.format(avg_prec, f1_score),
                 color=line_color)

        dec_prec = eval_infos[4]
        dec_recall = eval_infos[5]
        plt.plot(dec_recall, dec_prec, 'o', color=line_color, markersize=8)
        plt.vlines(dec_recall, 0, dec_prec, linestyles='dashed', colors=line_color)
        plt.hlines(dec_prec, 0, dec_recall, linestyles='dashed', colors=line_color)

    plt.legend(fontsize=basic_font_size)
    plt.title(plot_title, fontsize=basic_font_size+ 2)
    plt.xlabel('Recall', fontsize=basic_font_size)
    plt.ylabel('Precision', fontsize=basic_font_size)
    plt.xticks(fontsize=basic_font_size)
    plt.yticks(fontsize=basic_font_size)
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_multi_agg_pr_curves(line_name2pr_list, plot_title='Aggregated Precision-Recall Curve',
                             figsize=(12, 8), xlim=(0, 1), ylim=(0, 1), basic_font_size=14):
    plt.figure(figsize=figsize)

    for line_name, (prec_list, recall_list) in line_name2pr_list.items():
        plt.step(recall_list, prec_list, label=line_name)

    plt.legend(fontsize=basic_font_size)
    plt.title(plot_title, fontsize=basic_font_size+ 2)
    plt.xlabel('Recall', fontsize=basic_font_size)
    plt.ylabel('Precision', fontsize=basic_font_size)
    plt.xticks(fontsize=basic_font_size)
    plt.yticks(fontsize=basic_font_size)
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)


def get_gpu_mem_usage(gpu_id):
    gpu_qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    query_cmd = 'nvidia-smi -i {} --query-gpu={} --format=csv,noheader'.format(gpu_id, ','.join(gpu_qargs))
    pipe = os.popen(query_cmd)
    query_res = pipe.readlines()[0].strip('\n')
    items = query_res.split(',')
    mem_used = float(items[-2].strip(' MiB'))
    mem_total = float(items[-1].strip(' MiB'))
    return mem_used / mem_total


def wait_idle_gpu(gpu_id=None, mem_usage_ratio=0.01, sleep_second=2):
    if gpu_id is None:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
    print('{} Choose GPU {}, wait for memory usage ratio <= {}'.format(
        time.asctime(), gpu_id, mem_usage_ratio))
    sys.stdout.flush()
    while True:
        cur_mem_usage = get_gpu_mem_usage(gpu_id)
        if cur_mem_usage <= mem_usage_ratio:
            print('{} Current memory usage {:.5f}, start to bind gpu {}'.format(
                time.asctime(), cur_mem_usage, gpu_id
            ))
            apply_gpu_memory(gpu_id=0)
            break
        ss = random.randint(sleep_second, sleep_second + 20)
        time.sleep(ss)


def apply_gpu_memory(gpu_id=0):
    print('{} Choose gpu {}'.format(time.asctime(), os.environ['CUDA_VISIBLE_DEVICES']))
    # quickly apply a small part of gpu memory
    tmp_tensor = torch.FloatTensor(100, 100)
    cuda_device = 'cuda:{}'.format(gpu_id)
    tmp_tensor.to(cuda_device)


def set_all_random_seed(seed):
    print('Set random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
