# -*- coding: utf-8 -*-
# @Time    : 6/3/18 10:22
# @Author  : Shun Zheng

from __future__ import print_function

import os
import csv
import numpy as np
import pandas as pd
import torch
import torchtext.data as tt_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import average_precision_score, f1_score
import time

from utils import build_field_vocabulary, random_init_certain_vector, save_checkpoint, resume_checkpoint, \
    set_all_random_seed
from models import RelationLSTM, RelationAttBiLSTM, RelationSparseAttBiLSTM, BinarySoftNLLLoss, RelationPCNN
from optims import TruncateSGD, TruncateAdam


class RelationTask(object):
    def __init__(self, args, train_flag=True, dev_flag=True, test_flag=True, resume_flag=False, dump_flag=True):
        print('-' * 15, 'RelationTask Initialization', '-' * 15)
        if train_flag or dev_flag or test_flag:
            print('Setting Relation Task',
                  'train_flag =', train_flag,
                  'dev_flag =', dev_flag,
                  'test_flag =', test_flag)
        else:
            raise ValueError('train_flag, dev_flag, test_flag cannot be set to False at the same time')
        self._train_flag = train_flag
        self._dev_flag = dev_flag
        self._test_flag = test_flag
        self._dump_flag = dump_flag
        self._cur_best_avg_prec = 0.0
        self._cur_best_f1 = 0.0

        # get configuration dict
        self.raw_config = args.__dict__  # not modified
        self.config = dict(args.__dict__)  # modified later
        self.use_cuda = self.config['use_cuda']
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        self.fix_seq_len = None if self.config['fix_seq_len'] <= 0 else self.config['fix_seq_len']

        # -------------------------- Define Field object to process raw data --------------------------

        # Define sample id field
        self.ID = tt_data.Field(sequential=False,
                                use_vocab=False)

        # Define text field
        def normalize_named_entity(xs):
            for i, x_str in enumerate(xs):
                if '~' in x_str:
                    xs[i] = x_str.split('~')[0]

            return xs

        self.TEXT = tt_data.Field(sequential=True,
                                  fix_length=self.fix_seq_len,
                                  tokenize=lambda s: s.split(' '),
                                  preprocessing=normalize_named_entity,
                                  include_lengths=True)

        # Define position field
        pos_max_len = self.config['pos_max_len']

        def normalize_position_indicator(xs):
            for i, x in enumerate(xs):
                int_x = int(x)
                if int_x >= pos_max_len:
                    xs[i] = str(pos_max_len)
                elif int_x <= - pos_max_len:
                    xs[i] = str(-pos_max_len)

            return xs

        self.POS = tt_data.Field(sequential=True,
                                 fix_length=self.fix_seq_len,
                                 tokenize=lambda s: s.split(' '),
                                 preprocessing=normalize_position_indicator)

        # Define label field
        self.LABEL = tt_data.Field(sequential=False,
                                   use_vocab=False)
        self.train_label_type = self.config['train_label_type'].lower()
        if self.train_label_type == 'soft':
            self.TRAIN_LABEL = tt_data.Field(sequential=False,
                                             use_vocab=False,
                                             dtype=torch.float)
        else:
            self.TRAIN_LABEL = self.LABEL

        # -------------------------- Load dataset and dataset_iter objects --------------------------

        # load train set
        if self._train_flag:
            self.init_train_set()
        else:
            self.train_set = None
            self.train_iter = None

        # load dev set
        if self._dev_flag:
            self.init_dev_set()
        else:
            self.dev_set = None
            self.dev_iter = None

        # load test set
        if self._test_flag:
            self.init_test_set()
        else:
            self.test_set = None
            self.test_iter = None

        # for heldout evaluation purpose
        self.heldout_test_set = None
        self.heldout_entity_pairs = None
        self.heldout_test_iter = None

        # -------------------------- Build vocabulary --------------------------

        # build vocabulary for the TEXT field
        # If vocab_freq_file exists, build from it directly,
        # else, build by counting train_set and dump vocabulary statistics into vocab_freq_file
        print('Building word vocabulary dict')
        build_field_vocabulary(self.TEXT,
                               from_vocab=True,
                               vocab_freq_file=self.config['vocab_freq_file'],
                               data_set=self.train_set,
                               min_freq=self.config['min_word_freq'],
                               vectors=self.config['pretrained_word_vectors'])
        self.word_vocab = self.TEXT.vocab
        # random initialize the unknown token embedding
        random_init_certain_vector(self.word_vocab, self.TEXT.unk_token, mean=0, std=0.5)

        print('Building position vocabulary dict')
        # build vocabulary for the POS field by directly constructing a pseudo counter object
        pos_tokens = [str(x) for x in range(-pos_max_len, 0)] + ['0'] + [str(x) for x in range(1, pos_max_len + 1)]
        pos_freq_dict = dict(zip(pos_tokens, range(len(pos_tokens), 0, -1)))
        build_field_vocabulary(self.POS,
                               from_vocab=True,
                               vocab_freq_dict=pos_freq_dict)
        self.pos_vocab = self.POS.vocab

        # get the vocabulary size of words and positions
        word_vocab_size = len(self.word_vocab)
        pos_vocab_size = len(self.pos_vocab)
        self.config.update({
            'word_vocab_size': word_vocab_size,
            'pos_vocab_size': pos_vocab_size,
        })

        # -------------------------- Build neural networks --------------------------
        self.model_type = self.config['model_type']
        self.net = None
        self.loss_func = None
        self.optimizers = []
        self.init_neural_network()

        if resume_flag:
            self.resume_default_model(strict=False, resume_key='model')

        print('-' * 15, 'RelationTask Initialization End', '-' * 15)

    def init_train_set(self):
        set_all_random_seed(self.config['random_seed'])
        train_file_path = self.config['train_file']
        print('Loading train set from {}'.format(train_file_path))
        self.train_set = tt_data.TabularDataset(path=train_file_path,
                                                format='csv',
                                                fields=[('Id', self.ID),
                                                        ('Text', self.TEXT),
                                                        ('Pos1', self.POS),
                                                        ('Pos2', self.POS),
                                                        ('Label', self.TRAIN_LABEL)],
                                                skip_header=False)
        self.train_iter = tt_data.Iterator(self.train_set,
                                           sort_key=lambda x: len(x.Text),
                                           batch_size=self.config['train_batch_size'],
                                           train=True,
                                           repeat=False,
                                           sort_within_batch=True,
                                           device=self.device)

    def init_dev_set(self):
        dev_file_path = self.config['dev_file']
        print('Loading dev set from {}'.format(dev_file_path))
        self.dev_set = tt_data.TabularDataset(path=dev_file_path,
                                              format='csv',
                                              fields=[('Id', self.ID),
                                                      ('Text', self.TEXT),
                                                      ('Pos1', self.POS),
                                                      ('Pos2', self.POS),
                                                      ('Label', self.LABEL)],
                                              skip_header=False)
        self.dev_iter = tt_data.Iterator(self.dev_set,
                                         sort_key=lambda x: len(x.Text),
                                         batch_size=self.config['test_batch_size'],
                                         train=False,
                                         repeat=False,
                                         sort_within_batch=True,
                                         device=self.device)

    def init_test_set(self):
        test_file_path = self.config['test_file']
        print('Loading test set {}'.format(test_file_path))
        self.test_set = tt_data.TabularDataset(path=test_file_path,
                                               format='csv',
                                               fields=[('Id', self.ID),
                                                       ('Text', self.TEXT),
                                                       ('Pos1', self.POS),
                                                       ('Pos2', self.POS),
                                                       ('Label', self.LABEL)],
                                               skip_header=False)
        self.test_iter = tt_data.Iterator(self.test_set,
                                          sort_key=lambda x: len(x.Text),
                                          batch_size=self.config['test_batch_size'],
                                          train=False,
                                          repeat=False,
                                          sort_within_batch=True,
                                          device=self.device)

    def init_heldout_test_set(self):
        # TODO: change this into input arguments
        data_dir_path = os.path.dirname(self.config['test_file'])
        heldout_test_file_path = os.path.join(data_dir_path, 'nyt_heldout_test.csv')
        heldout_test_entitypair_fp = os.path.join(data_dir_path, 'nyt_heldout_test_entitypair.csv')

        def read_entity_pair_info(entitypair_file_path):
            tmp_df = pd.read_csv(entitypair_file_path, header=None)
            tmp_df.columns = ['span1_guid', 'span2_guid', 'span1', 'span2']
            entitypair_infos = tmp_df.to_dict(orient='records')
            entity_pairs = []
            for ep_info in entitypair_infos:
                entity_pairs.append((ep_info['span1_guid'], ep_info['span2_guid']))

            return entity_pairs

        print('Loading heldout test set {}'.format(heldout_test_file_path))
        self.heldout_test_set = tt_data.TabularDataset(path=heldout_test_file_path,
                                                       format='csv',
                                                       fields=[('Id', self.ID),
                                                               ('Text', self.TEXT),
                                                               ('Pos1', self.POS),
                                                               ('Pos2', self.POS),
                                                               ('Label', self.LABEL)],
                                                       skip_header=False)
        self.heldout_entity_pairs = read_entity_pair_info(heldout_test_entitypair_fp)
        self.heldout_test_iter = tt_data.Iterator(self.heldout_test_set,
                                                  sort_key=lambda x: len(x.Text),
                                                  batch_size=self.config['test_batch_size'],
                                                  train=False,
                                                  repeat=False,
                                                  sort_within_batch=True,
                                                  device=self.device)

    def init_neural_network(self):
        set_all_random_seed(self.config['random_seed'])
        self.net = self.get_neural_network(self.model_type, self.device)
        self.loss_func = self.get_loss_func()
        self.optimizers = self.get_optimizers(self.net.parameters())
        self._cur_best_avg_prec = 0.0

    def get_neural_network(self, model_type, device):
        if model_type == 'LSTM':
            net = RelationLSTM(self.config['word_vocab_size'],
                               self.config['word_vec_size'],
                               self.config['pos_vocab_size'],
                               self.config['pos_vec_size'],
                               self.config['hidden_size'],
                               self.config['class_size'],
                               pre_word_vecs=self.word_vocab.vectors,
                               use_cuda=self.use_cuda)
        elif model_type == 'AttBiLSTM':
            net = RelationAttBiLSTM(self.config['word_vocab_size'],
                                    self.config['word_vec_size'],
                                    self.config['pos_vocab_size'],
                                    self.config['pos_vec_size'],
                                    self.config['hidden_size'],
                                    self.config['class_size'],
                                    pre_word_vecs=self.word_vocab.vectors,
                                    emb_dropout_p=self.config['emb_dropout_p'],
                                    lstm_dropout_p=self.config['lstm_dropout_p'],
                                    last_dropout_p=self.config['last_dropout_p'])
        elif model_type == 'SparseAttBiLSTM':
            net = RelationSparseAttBiLSTM(self.config['word_vocab_size'],
                                          self.config['word_vec_size'],
                                          self.config['pos_vocab_size'],
                                          self.config['pos_vec_size'],
                                          self.config['hidden_size'],
                                          self.config['class_size'],
                                          pre_word_vecs=self.word_vocab.vectors,
                                          emb_dropout_p=self.config['emb_dropout_p'],
                                          lstm_dropout_p=self.config['lstm_dropout_p'],
                                          last_dropout_p=self.config['last_dropout_p'])
        elif model_type == 'PCNN':
            net = RelationPCNN(self.config['word_vocab_size'],
                               self.config['word_vec_size'],
                               self.config['pos_vocab_size'],
                               self.config['pos_vec_size'],
                               self.config['class_size'],
                               self.pos_vocab.stoi['0'],
                               filter_size=230,
                               window_size=3,
                               pre_word_vecs=self.word_vocab.vectors,
                               last_dropout_p=self.config['last_dropout_p'])
        else:
            raise ValueError('Unsupported neural network type {}'.format(model_type))
        net.to(device)  # this method modified the module in-place

        return net

    def get_loss_func(self):
        if self.train_label_type == 'hard':
            print('Choose negative log likelihood loss')
            return nn.NLLLoss()
        elif self.train_label_type == 'soft':
            print('Choose negative log likelihood loss with soft labels')
            return BinarySoftNLLLoss()
        else:
            raise ValueError('Unsupported label type: {}'.format(self.train_label_type))

    def get_optimizers(self, parameters):
        opti_type = self.config['optimizer_type']

        # temporal code to experiment sparse mask embedding
        if self.model_type == 'SparseAttBiLSTM':
            optim_list = []
            if opti_type.lower() == 'truncatesgd':
                mask_optim = TruncateSGD(self.net.get_parameters(mask_flag=True),
                                         lr=self.config['lr_truncate'],
                                         gravity=self.config['gravity'],
                                         truncate_freq=self.config['truncate_freq'])
                print('Optimize mask parameters with TruncateSGD')
                optim_list.append(mask_optim)
            elif opti_type.lower() == 'truncateadam':
                mask_optim = TruncateAdam(self.net.get_parameters(mask_flag=True),
                                          lr_truncate=self.config['lr_truncate'],
                                          gravity=self.config['gravity'],
                                          truncate_freq=self.config['truncate_freq'],
                                          lr=self.config['learning_rate'],
                                          weight_decay=self.config['weight_decay'])
                print('Optimize mask parameters with TruncateAdam')
                optim_list.append(mask_optim)
            else:
                raise ValueError('Unsupported optimizer type:', opti_type, 'for mask parameter')

            if self.config['optimize_parameters'].lower() == 'all':
                non_mask_optim = optim.Adam(self.net.get_parameters(non_mask_flag=True),
                                            lr=self.config['learning_rate'],
                                            weight_decay=self.config['weight_decay'])
                print('Optimize non-mask parameters with Adam')
                optim_list.append(non_mask_optim)
            elif self.config['optimize_parameters'].lower() == 'mask':
                self.net.only_train_mask_parameter()
                print('Do not train non-mask parameters')
            else:
                self.net.only_train_mask_parameter()
                print('Do not train non-mask parameters')

            if self.config['mask_init_weight'].lower() == 'one':
                print('Initialize mask embedding to one')
                self.net.init_mask_to_one()
            else:
                print('Initialize mask embedding randomly')

            return optim_list

        print('Choose {} optimizer'.format(opti_type.lower()))
        if opti_type.lower() == 'sgd':
            optimizer = optim.SGD(parameters,
                                  lr=self.config['learning_rate'],
                                  momentum=self.config['momentum'],
                                  weight_decay=self.config['weight_decay'])
        elif opti_type.lower() == 'adagrad':
            optimizer = optim.Adagrad(parameters,
                                      lr=self.config['learning_rate'],
                                      weight_decay=self.config['weight_decay'])
        elif opti_type.lower() == 'adadelta':
            optimizer = optim.Adadelta(parameters,
                                       lr=self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])
        elif opti_type.lower() == 'adam':
            optimizer = optim.Adam(parameters,
                                   lr=self.config['learning_rate'],
                                   weight_decay=self.config['weight_decay'])
        else:
            raise ValueError('Unsupported optimizer type:', opti_type)

        return [optimizer]

    @staticmethod
    def prepare_relation_mini_batch(device, mini_batch):
        # prepare a mini-batch of data to train for the relation extraction task
        input_ids = mini_batch.Id.to(device)
        input_words = mini_batch.Text[0].to(device)
        input_lengths = mini_batch.Text[1].to(device)
        input_pos1s = mini_batch.Pos1.to(device)
        input_pos2s = mini_batch.Pos2.to(device)
        input_labels = mini_batch.Label.to(device)
        input_batch = (input_words, input_pos1s, input_pos2s)

        return input_ids, input_batch, input_lengths, input_labels

    def resume_default_model(self, strict=False, resume_key='model'):
        model_file_path = os.path.join(self.config['model_dir'],
                                       self.config['model_resume_name'])
        resume_checkpoint(self.net, model_file_path, strict=strict, resume_key=resume_key)

    def resume_model_from(self, model_file_path, strict=False, resume_key='model'):
        resume_checkpoint(self.net, model_file_path, strict=strict, resume_key=resume_key)

    def validate_dev_avg_prec(self):
        cur_best_avg_prec = self._cur_best_avg_prec

        if self._dev_flag:
            dev_avg_prec = self.get_average_precision_score(self.dev_iter)
            print('*' * 5, 'Current dev average precision score', dev_avg_prec)

            if dev_avg_prec > cur_best_avg_prec:
                self._cur_best_avg_prec = dev_avg_prec
                is_best_flag = True
                print('*' * 5, 'Get new best dev average precision score', dev_avg_prec)
            else:
                is_best_flag = False
                print('*' * 5, 'Current best dev average precision score', cur_best_avg_prec)
        else:
            dev_avg_prec = None
            is_best_flag = True

        cur_state = {
            'dev_avg_prec': dev_avg_prec,
            'model': self.net.state_dict()
        }

        return cur_state, is_best_flag

    def validate_dev_f1(self):
        cur_best_f1 = self._cur_best_f1

        if self._dev_flag:
            dev_f1 = self.get_f1_score(self.dev_iter)
            print('*' * 5, 'Current dev f1 score', dev_f1)

            if dev_f1 > cur_best_f1:
                self._cur_best_f1 = dev_f1
                is_best_flag = True
                print('*' * 5, 'Get new best dev f1 score', dev_f1)
            else:
                is_best_flag = False
                print('*' * 5, 'Current best f1 score', cur_best_f1)
        else:
            dev_f1 = None
            is_best_flag = True

        cur_state = {
            'dev_f1': dev_f1,
            'model': self.net.state_dict()
        }

        return cur_state, is_best_flag

    def dump_best_model(self, epoch, model_state, is_best_flag=True):
        # Dump model state
        if self._dump_flag:
            if not os.path.exists(self.config['model_dir']):
                os.mkdir(self.config['model_dir'])
            model_file_path_prefix = os.path.join(self.config['model_dir'],
                                                  self.config['model_store_name_prefix'])
            tmp_suffix = '.' + str(epoch)
            model_state['epoch'] = epoch
            save_checkpoint(model_state, is_best_flag, model_file_path_prefix, tmp_suffix)
            print('*' * 5, 'Dump model into', model_file_path_prefix + tmp_suffix)

    def train(self, valid_callback=None, max_epoch=None, print_loss_freq=None):
        if not self._train_flag:
            raise AttributeError('When calling train(), _train_flag needs to be set as True')
        set_all_random_seed(self.config['random_seed'])

        if print_loss_freq is None:
            print_loss_freq = self.config['print_loss_freq']
        if max_epoch is None:
            max_epoch = self.config['max_epoch']

        # iterate through the data
        for epoch in range(max_epoch + 1):
            if epoch > 0:  # exit condition
                # enter inference mode
                print('[ Epoch', epoch, 'eval on dev set]')
                self.net.eval()
                if valid_callback is None:
                    # run until the last epoch and treat it as the best
                    cur_state = {
                        'model': self.net.state_dict()
                    }
                    is_best_flag = True
                else:
                    cur_state, is_best_flag = valid_callback(self)
                self.dump_best_model(epoch, cur_state, is_best_flag)
                if epoch == max_epoch:
                    break

            # enter train mode
            self.net.train()
            print('[ Epoch', epoch + 1, 'starts ]')
            for idx, mini_batch in enumerate(self.train_iter):
                # prepare mini-batch
                _, input_batch, input_lengths, input_labels = \
                    self.prepare_relation_mini_batch(self.device, mini_batch)

                # forward process
                out_log_probs = self.net(input_batch, input_lengths)
                loss = self.loss_func(out_log_probs, input_labels)

                # backward process
                self.net.zero_grad()
                loss.backward()
                for opt in self.optimizers:
                    opt.step()

                if print_loss_freq > 0 and idx % print_loss_freq == 0:
                    print(time.asctime(), 'Mini-batch', idx, 'loss', loss.item())
            print('[ Epoch', epoch + 1, 'ends ]')

    def eval_dev(self, acc_flag=True, avg_prec_flag=True):
        if not self._dev_flag:
            raise AttributeError('When calling eval_dev(), _dev_flag needs to be set as True')

        if acc_flag:
            dev_acc = self.get_prediction_accuracy(self.dev_iter)
            print('[ Dev Set Predict Accuracy', dev_acc, ']')

        if avg_prec_flag:
            dev_avg_prec = self.get_average_precision_score(self.dev_iter, average='macro')
            print('[ Dev Set Predict Average Precision', dev_avg_prec, ']')

    def eval_test(self, acc_flag=True, avg_prec_flag=True):
        if not self._test_flag:
            raise AttributeError('When calling eval_test(), _test_flag needs to be set as True')

        if acc_flag:
            test_acc = self.get_prediction_accuracy(self.test_iter)
            print('[ Test Set Predict Accuracy', test_acc, ']')

        if avg_prec_flag:
            test_avg_prec = self.get_average_precision_score(self.test_iter, average='macro')
            print('[ Test Set Predict Average Precision', test_avg_prec, ']')

    def eval_heldout_test(self):
        print('Heldout Evaluation:')
        if self.heldout_test_set is None or self.heldout_test_iter is None or self.heldout_entity_pairs is None:
            self.init_heldout_test_set()

        # build example id to example id dictionary
        example_id2entity_pair = {}
        for entity_pair, example in zip(self.heldout_entity_pairs, self.heldout_test_set.examples):
            example_id = int(example.Id)
            example_id2entity_pair[example_id] = entity_pair

        example_ids, class_pred_probs, true_labels = self.get_prediction_probs_info(self.heldout_test_iter, in_cpu=True)
        _, pred_labels = torch.max(class_pred_probs, dim=-1)
        example_ids = example_ids.tolist()
        class_pred_probs = class_pred_probs.tolist()
        pred_labels = pred_labels.tolist()
        true_labels = true_labels.tolist()

        entity_pair2gold_label_set = {}
        entity_pair2eid_pred_label_prob = {}
        for example_id, class_pred_prob, pred_label, true_label in zip(example_ids, class_pred_probs, pred_labels,
                                                                       true_labels):
            entity_pair = example_id2entity_pair[example_id]

            # fill entity pair -> gold labels dictionary
            if true_label != 0:
                if entity_pair in entity_pair2gold_label_set:
                    entity_pair2gold_label_set[entity_pair].add(true_label)
                else:
                    entity_pair2gold_label_set[entity_pair] = set()
                    entity_pair2gold_label_set[entity_pair].add(true_label)

            # fill entity pair -> maximum prediction prob, label
            # error code:
            # if true_label != 0 or pred_label != 0:  # not NA relation
            if pred_label != 0:  # not NA relation
                # if True:
                # TODO: consider multi-label version
                if entity_pair not in entity_pair2eid_pred_label_prob:
                    entity_pair2eid_pred_label_prob[entity_pair] = (example_id, pred_label, class_pred_prob[pred_label])
                else:
                    _, prev_label, prev_label_prob = entity_pair2eid_pred_label_prob[entity_pair]
                    if class_pred_prob[pred_label] >= prev_label_prob:
                        entity_pair2eid_pred_label_prob[entity_pair] = (
                        example_id, pred_label, class_pred_prob[pred_label])

        num_rel_entity_pair = 0
        for entity_pair, gold_labels in entity_pair2gold_label_set.items():
            if 0 not in gold_labels:
                num_rel_entity_pair += 1

        entity_pair_pred_label_probs = []
        for entity_pair, eid_pred_label_prob_tuple in entity_pair2eid_pred_label_prob.items():
            entity_pair_pred_label_probs.append(
                (entity_pair, eid_pred_label_prob_tuple[1], eid_pred_label_prob_tuple[2]))
        entity_pair_pred_label_probs.sort(key=lambda x: x[-1], reverse=True)

        prec_at_n = []
        prec_list = []
        recall_list = []
        tp = 0.0
        for idx, items in enumerate(entity_pair_pred_label_probs):
            if 0 < idx <= 300 and idx % 100 == 0:
                prec_at_n.append((tp / idx, idx))
            if idx > 0 and idx % 10 == 0:
                prec_list.append(tp / idx)
                recall_list.append(tp / num_rel_entity_pair)

            entity_pair = items[0]
            pred_label = items[1]
            if entity_pair in entity_pair2gold_label_set and pred_label in entity_pair2gold_label_set[entity_pair]:
                tp += 1

        tmp_prec = tp / len(entity_pair_pred_label_probs)
        tmp_recall = tp / num_rel_entity_pair
        if int(tp) != 0:
            tmp_f1 = 2.0 / (1.0 / tmp_prec + 1.0 / tmp_recall)
        else:
            tmp_f1 = 0.0
        print('{} entity pairs in total, {} has gold relations, {} relation predictions'.format(
            len(entity_pair2gold_label_set), num_rel_entity_pair, len(entity_pair_pred_label_probs)))
        print('Prec: {}, Recall: {}, F1: {}, Prec@N: {}'.format(tmp_prec, tmp_recall, tmp_f1, prec_at_n))

        pred_info = {
            'entity_pair2gold_label_set': entity_pair2gold_label_set,
            'entity_pair2eid_pred_label_prob': entity_pair2eid_pred_label_prob,
        }

        return prec_list, recall_list, pred_info

    def get_prediction_class_info(self, dataset_iter, in_cpu=True, list_format=False):
        self.net.eval()  # enter the evaluation model first
        raw_ids = []
        pred_cids = []
        true_cids = []

        for mini_batch in dataset_iter:
            # prepare mini-batch
            input_ids, input_batch, input_lengths, input_labels = \
                self.prepare_relation_mini_batch(self.device, mini_batch)
            # forward process
            with torch.no_grad():
                out_log_probs = self.net(input_batch, input_lengths)
            _, out_class_ids = torch.max(out_log_probs, dim=-1)

            mb_raw_ids = input_ids.data  # raw instance id
            mb_pred_cids = out_class_ids.data  # predicted class id
            mb_true_cids = input_labels.data  # true class id
            raw_ids.append(mb_raw_ids)
            pred_cids.append(mb_pred_cids)
            true_cids.append(mb_true_cids)
        raw_ids = torch.cat(raw_ids, dim=0)
        pred_cids = torch.cat(pred_cids, dim=0)
        true_cids = torch.cat(true_cids, dim=0)
        if in_cpu:
            raw_ids, pred_cids, true_cids = raw_ids.cpu(), pred_cids.cpu(), true_cids.cpu()
            # tolist() can only be called after .cpu() operation
            if list_format:
                raw_ids, pred_cids, true_cids = raw_ids.tolist(), pred_cids.tolist(), true_cids.tolist()

        return raw_ids, pred_cids, true_cids

    def get_prediction_accuracy(self, dataset_iter):
        raw_ids, pred_cids, true_cids = self.get_prediction_class_info(dataset_iter)
        num_sample = raw_ids.size(0)
        result = pred_cids == true_cids
        acc = float(torch.sum(result)) / num_sample

        return acc

    def get_prediction_probs_info(self, dataset_iter, in_cpu=True, list_format=False):
        self.net.eval()  # enter the evaluation model first
        raw_ids = []
        pred_probs = []
        true_cids = []
        for mini_batch in dataset_iter:
            # prepare mini-batch
            input_ids, input_batch, input_lengths, input_labels = \
                self.prepare_relation_mini_batch(self.device, mini_batch)
            # forward process
            with torch.no_grad():
                out_log_probs = self.net(input_batch, input_lengths)
            out_probs = torch.exp(out_log_probs)

            mb_raw_ids = input_ids.data  # raw instance id
            mb_pred_probs = out_probs.data  # predicted probabilities
            mb_true_cids = input_labels.data  # true class id
            raw_ids.append(mb_raw_ids)
            pred_probs.append(mb_pred_probs)
            true_cids.append(mb_true_cids)
        raw_ids = torch.cat(raw_ids, dim=0).cpu()
        pred_probs = torch.cat(pred_probs, dim=0).cpu()
        true_cids = torch.cat(true_cids, dim=0).cpu()
        if in_cpu:
            raw_ids, pred_probs, true_cids = raw_ids.cpu(), pred_probs.cpu(), true_cids.cpu()
            # tolist() can only be called after .cpu() operation
            if list_format:
                raw_ids, pred_probs, true_cids = raw_ids.tolist(), pred_probs.tolist(), true_cids.tolist()

        return raw_ids, pred_probs, true_cids

    def get_average_precision_score(self, dataset_iter, average='macro'):
        _, pred_probs, true_cids = self.get_prediction_probs_info(dataset_iter)
        num_samples, num_classes = pred_probs.size()
        true_labels = torch.zeros(num_samples, num_classes).long()
        batch_idxs = torch.arange(num_samples).long()
        true_labels[batch_idxs, true_cids] = 1
        avg_prec = average_precision_score(true_labels[:, 1:].numpy(),
                                           pred_probs[:, 1:].numpy(),
                                           average=average)
        return avg_prec

    def get_f1_score(self, dataset_iter, average='macro'):
        _, pred_probs, true_cids = self.get_prediction_probs_info(dataset_iter)
        num_samples, num_classes = pred_probs.size()
        true_labels = torch.zeros(num_samples, num_classes).long()
        batch_idxs = torch.arange(num_samples).long()
        true_labels[batch_idxs, true_cids] = 1
        lid = 1
        pred_thresh = 0.5
        cur_f1_score = f1_score(true_labels[:, lid].numpy(),
                                (pred_probs[:, lid] > pred_thresh).numpy())
        return cur_f1_score

    def get_decompose_info(self, dataset_iter, decompose_type='beta', sorted_by_raw_id=False):
        if self.model_type != 'LSTM':
            raise ValueError('Only LSTM network can get decompose information, while the network is', self.model_type)

        all_raw_ids = []
        all_word_ids = []
        all_decompose_vals = []
        all_true_cids = []
        for mini_batch in dataset_iter:
            # prepare mini-batch
            input_ids, input_batch, input_lengths, input_labels = \
                self.prepare_relation_mini_batch(self.device, mini_batch)
            if isinstance(input_batch, tuple):
                input_words = input_batch[0]
            else:
                input_words = input_batch
            # get decompose values
            with torch.no_grad():
                if decompose_type == 'beta':
                    out_decompose, _ = self.net.decompose(input_batch, input_lengths)
                elif decompose_type == 'gamma':
                    out_decompose, _ = self.net.additive_decompose(input_batch, input_lengths)
                else:
                    raise ValueError('Unsupported decomposition type', decompose_type)
            for bid in range(mini_batch.batch_size):
                all_raw_ids.append(input_ids.data[bid])
                seq_len = input_lengths[bid]
                all_word_ids.append(input_words[:seq_len, bid].data.tolist())
                all_decompose_vals.append(out_decompose[:seq_len, bid, :].data.tolist())
                all_true_cids.append(input_labels.data[bid])
        decompose_info = zip(all_raw_ids, all_word_ids, all_decompose_vals, all_true_cids)
        if sorted_by_raw_id:
            decompose_info = sorted(decompose_info, key=lambda x: x[0])

        return decompose_info

    def dump_prediction_info(self, dataset_iter, file_name=None, pred_thresh=0.5):
        raw_ids, pred_probs, true_cids = self.get_prediction_probs_info(dataset_iter)
        num_samples, num_classes = pred_probs.size()

        # calculate average precision score
        true_labels = torch.zeros(num_samples, num_classes).long()
        batch_idxs = torch.arange(num_samples).long()
        true_labels[batch_idxs, true_cids] = 1
        avg_prec = average_precision_score(true_labels[:, 1:].numpy(),
                                           pred_probs[:, 1:].numpy(),
                                           average='macro')

        print('-' * 15, 'Prediction Statistics', '-' * 15)
        print('Total average precision score:', avg_prec)
        for lid in range(1, num_classes):
            tmp_avg_prec = average_precision_score(true_labels[:, lid].numpy(),
                                                   pred_probs[:, lid].numpy(),
                                                   average='macro')
            tmp_f1_score = f1_score(true_labels[:, lid].numpy(),
                                    (pred_probs[:, lid] > pred_thresh).numpy())
            print('Class {}, average score:{}, f1 score:{}'.format(lid, tmp_avg_prec, tmp_f1_score))

        if file_name is not None:
            file_path = os.path.join(self.config['model_dir'], file_name)
            print('Dumping prediction information into', file_path)
            with open(file_path, 'w') as fout:
                csv_writer = csv.writer(fout)
                header = ['Id', 'Label'] + ['Class' + str(i) for i in range(num_classes)]
                csv_writer.writerow(header)
                for idx in range(num_samples):
                    row_list = [raw_ids[idx], true_cids[idx]] + list(pred_probs[idx])
                    csv_writer.writerow(row_list)
