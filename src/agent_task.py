# -*- coding: utf-8 -*-
# @Time    : 3/5/18 10:28
# @Author  : Shun Zheng

from __future__ import print_function

import os
import time
import random
from collections import namedtuple
import numpy as np
import math
import pickle
from torchtext.data.batch import Batch
import torchtext.data as tt_data
import torch
import torch.optim as optim
import torch.distributions as dist
from torch.distributions.utils import probs_to_logits

from relation_task import RelationTask
from models import ErasurePolicy
from utils import resume_checkpoint, save_checkpoint
from rule_helpers import get_decision_rules, get_rule_quality, merge_common_rules
from label_factory import create_label_func, LabelFactory

TempBatch = namedtuple('TempBatch', ('Id', 'Text', 'Pos1', 'Pos2', 'Label'))


class RelationEnv(RelationTask):
    """
    Simulate the relation classification neural network as an environment for rl agent.
    """

    def __init__(self, args):
        print('-' * 15, 'RelationEnv Initialization', '-' * 15)
        super(RelationEnv, self).__init__(args, train_flag=True, dev_flag=True, test_flag=True, resume_flag=True)

        self.raw_train_examples = self.train_set.examples
        self.raw_dev_examples = self.dev_set.examples
        self.raw_test_examples = self.test_set.examples
        self.env_data_iter = None
        self.env_data_set = None
        self.set_data_iter(data_type='train', train_mode=True)

        # a hyper-parameter to weight erasure reward
        self.reward_eta = self.config['policy_reward_eta']
        # enter inference mode

        self.state_size = self.net.input_emb_size
        # old version 2
        # # input embedding + previous hidden semantic embedding
        # self.semantic_emb_size = self.net.hidden_size * 2
        # self.state_size = self.net.input_emb_size + self.semantic_emb_size

        # state relevant variables
        self.step_index = -1
        self.step_length = -1

        self.raw_example_ids = None
        self.raw_example_batch = None
        self.input_emb_seq = None
        self.input_seq_lens = None
        self.raw_tgt_logp = None
        self.raw_pred_logps = None

        self.forward_lstm_state = None  # (hx, cx)
        self.backward_lstm_emb = None

        self.current_pred_logps = None
        self.current_batch = None
        self.prev_actions = []

        print('-' * 15, 'RelationEnv Initialization End', '-' * 15)

    def filter_environment_train_set(self, pred_filter_prob=0.5):
        # resume original train examples
        self.train_set.examples = self.raw_train_examples
        # train_iter will iterate over all train examples
        raw_ids, pred_probs, _ = self.get_prediction_probs_info(self.train_iter, in_cpu=True, list_format=True)

        if pred_filter_prob < 1.0:
            id2pred_probs = dict(zip(raw_ids, pred_probs))
            # obtain filtered examples
            filtered_examples = []
            for ex in self.raw_train_examples:
                eid = int(ex.Id)
                if eid in id2pred_probs and id2pred_probs[eid][1] >= pred_filter_prob:
                    filtered_examples.append(ex)
            # set filtered examples to train set
            self.train_set.examples = filtered_examples
        else:
            pred_filter_prob = int(pred_filter_prob)
            assert pred_filter_prob > 100
            num_retain_examples = pred_filter_prob
            print('Dynamically change pred filter prob to preserve {} examples'.format(num_retain_examples))

            example_id2example = {}
            for ex in self.train_set.examples:
                example_id2example[int(ex.Id)] = ex

            eid_prob_list = [(raw_id, pred_prob_arr[1]) for raw_id, pred_prob_arr in zip(raw_ids, pred_probs)]
            eid_prob_list.sort(key=lambda x: x[1], reverse=True)
            pred_filter_prob = eid_prob_list[num_retain_examples-1][1]
            filtered_examples = []
            for idx in range(num_retain_examples):
                eid, prob = eid_prob_list[idx]
                filtered_examples.append(example_id2example[eid])
            # set filtered examples to train set
            self.train_set.examples = filtered_examples

        print('Filter training examples by the prediction probability {0:.2f}, reduce from {1} to {2}'.format(
            pred_filter_prob, len(self.raw_train_examples), len(self.train_set.examples)))

    def set_data_iter(self, data_type='train', train_mode=True, batch_size=1):
        if data_type == 'train':
            arg_data_set = self.train_set
        elif data_type == 'dev':
            arg_data_set = self.dev_set
        elif data_type == 'test':
            arg_data_set = self.test_set
        else:
            raise ValueError('Unsupported data_type value {}, must be in [train, dev, test]'.format(data_type))

        if train_mode:
            arg_repeat, arg_shuffle, arg_sort = True, True, False
        else:
            arg_repeat, arg_shuffle, arg_sort = False, False, False

        # note that batch_size is set to 1
        self.env_data_iter = iter(tt_data.Iterator(arg_data_set, batch_size=batch_size, sort_key=lambda x: len(x.Text),
                                                   repeat=arg_repeat, shuffle=arg_shuffle, sort=arg_sort,
                                                   sort_within_batch=True, device=self.device))
        self.env_data_set = arg_data_set

        print("Set environment data iterator, data_type='{}', train_mode={}, batch_size={}".format(
            data_type, train_mode, batch_size))

    def set_reward_eta(self, reward_eta):
        print('Set reward_eta to {} (last value {})'.format(reward_eta, self.reward_eta))
        self.reward_eta = reward_eta

    def next_example(self, example=None):
        # the mini-batch only contains one example due to batch_size == 1
        if example is None:
            self.raw_example_batch = next(self.env_data_iter)
        else:
            self.raw_example_batch = Batch(data=[example], dataset=self.train_set, device=self.device)

        # reserve example ids to track back to data sources
        self.raw_example_ids = self.raw_example_batch.Id.tolist()

        # get raw prediction probability
        _, input_batch, input_lengths, _ = self.prepare_relation_mini_batch(self.device, self.raw_example_batch)
        self.net.eval()
        with torch.no_grad():
            input_emb_seq, pred_logp = self.net.get_init_state_info(input_batch, input_lengths)

        # set input embedding (word embedding + position embedding)
        self.input_emb_seq = input_emb_seq
        self.input_seq_lens = input_lengths

        # set raw prediction log-probability for all classes
        self.raw_tgt_logp = pred_logp[:, 1]
        self.raw_pred_logps = pred_logp.tolist()

        # old version 2
        # # set backward lstm embedding, seq_len x batch_size x hidden_size
        # self.backward_lstm_emb = lstm_output[:, :, self.net.hidden_size:]

    def reset_state(self):
        # get an initial word index copy
        # init_words = torch.tensor(self.raw_example_batch.Text[0])
        # for torch 1.1.0
        init_words = self.raw_example_batch.Text[0].clone().detach()
        # init_lengths = torch.zeros_like(self.raw_example_batch.Text[1])
        init_lengths = self.raw_example_batch.Text[1]
        # reset current example batch
        # Note: this batch object will be modified at each step according to the action
        self.current_batch = TempBatch(self.raw_example_batch.Id,
                                       (init_words, init_lengths),
                                       self.raw_example_batch.Pos1,
                                       self.raw_example_batch.Pos2,
                                       self.raw_example_batch.Label)
        self.current_pred_logps = None

    def create_new_batch(self):
        # get an initial word index copy
        init_words = torch.tensor(self.raw_example_batch.Text[0])
        # init_lengths = torch.zeros_like(self.raw_example_batch.Text[1])
        init_lengths = self.raw_example_batch.Text[1]
        # reset current example batch
        # Note: this batch object will be modified at each step according to the action
        temp_batch = TempBatch(self.raw_example_batch.Id,
                               (init_words, init_lengths),
                               self.raw_example_batch.Pos1,
                               self.raw_example_batch.Pos2,
                               self.raw_example_batch.Label)

        return temp_batch

    def batch_transition_with(self, seq_action_masks):
        # seq_action_masks.shape is Size(seq_len, batch_size, action_size)

        # set input text according to erasure actions
        seq_erase_actions = seq_action_masks[:, :, 1]  # 1: erase 0: retain, Size(seq_len, batch_size)
        self.current_batch.Text[0][seq_erase_actions == 1] = 1  # 1 denote <pad> token

        # get reward
        batch_llh_reward = self.get_batch_llh_reward()
        batch_erase_reward = self.get_batch_erase_reward(seq_erase_actions)
        batch_total_reward = batch_llh_reward + self.reward_eta * batch_erase_reward  # Size(batch_size)

        return batch_total_reward

    def get_batch_llh_reward(self):
        _, input_batch, input_lengths, _ = self.prepare_relation_mini_batch(self.device, self.current_batch)
        with torch.no_grad():
            batch_pred_logp = self.net(input_batch, input_lengths)  # Size(batch_size, class_size)
            batch_tgt_logp = batch_pred_logp[:, 1]  # 1 denotes target relation index, Size(batch_size)
            batch_llh_reward = batch_tgt_logp - self.raw_tgt_logp  # Size([batch_size])
        self.current_pred_logps = batch_pred_logp.tolist()

        return batch_llh_reward

    def get_batch_erase_reward(self, seq_erase_actions):
        # seq_erase_actions.shape is Size(seq_len, batch_size), 1: erase, 0: retain
        batch_seq_lens = self.input_seq_lens.float()  # Size(batch_size), dtype=torch.float
        batch_erase_ratios = seq_erase_actions.sum(dim=0) / batch_seq_lens  # Size(batch_size)

        return batch_erase_ratios

    def transition_with(self, action):
        # apply action at current step
        if action == 1:
            # 1: erasure the word at the current step, 0: keep original word at the current step
            self.current_batch.Text[0][self.step_index, 0] = 1  # 1 is the index of '<pad>'
        # self.current_batch.Text[1][0] = self.step_index + 1  # update current lengths
        self.prev_actions.append(action)  # collect history actions

        # old version 2
        # self.update_forward_lstm_state(self.step_index)  # update forward lstm state

        # move to the next step
        self.step_index += 1
        reward = self.get_reward_at(self.step_index)
        next_state = self.get_state_embedding_at(self.step_index)
        done = True if self.step_index >= self.step_length else False

        return reward, next_state, done

    # TODO: how to design a better reward strategy
    def get_reward_at(self, next_step):
        """
        Get the reward when transiting from (step, action) to (next_step, )

        Args:
            next_step: the index of the next step

        Returns:
            reward for the transition of (step, action) -> (next_step, )
        """
        if next_step == self.step_length:
            assert self.current_batch.Text[1][0].item() == self.step_length
            llh_reward = self.get_likelihood_reward()
            del_reward = self.get_irrelevant_erasure_reward()
            total_reward = llh_reward + self.reward_eta * del_reward
            # print('llh reward ({}) + eta ({}) * del reward ({}) = total reward ({})'.format(
            #     llh_reward, self.reward_eta, del_reward, total_reward))
            return total_reward
        return 0

    def get_state_embedding_at(self, step):
        """
        Define how to get state embedding for each step

        Args:
            step: an integer to indicate current step index

        Returns:
            state embedding for this step index
        """
        if 0 <= step < self.step_length:
            state_emb = self.input_emb_seq[step]  # 1 x input_emb_size
        else:
            state_emb = self.input_emb_seq.new_zeros((1, self.net.input_emb_size))
        assert state_emb.requires_grad is False

        return state_emb

    def get_likelihood_reward(self):
        """
        Assume two classes: 'NA' and 'Target', the reward is equal to
        the log probability of 'Target' class minus raw log probability
        """
        _, input_batch, input_lengths, _ = self.prepare_relation_mini_batch(self.device, self.current_batch)
        with torch.no_grad():
            pred_logp = self.net(input_batch, input_lengths)  # size([1, class_size])
        self.current_pred_logps = pred_logp.tolist()
        target_logp = self.current_pred_logps[0][1]
        # raw_target_logp = self.raw_pred_logps[0][1]
        # llh_reward = target_logp - raw_target_logp
        llh_reward = target_logp

        return llh_reward

    def get_irrelevant_erasure_reward(self):
        """
        Calculate reward for erasing irrelevant words
        """
        assert len(self.prev_actions) == self.step_length
        erasure_num = float(sum(self.prev_actions))

        return erasure_num / self.step_length

    def get_semantic_embedding_at(self, step):
        """
        Calculate semantic embedding for sequences [0, ..., step-1] that represent historical information

        Args:
            step: an integer to indicate current step index

        Returns:
            semantic embedding for this step index
        """
        input_words = self.current_batch.Text[0][:step]
        input_pos1s = self.current_batch.Pos1[:step]
        input_pos2s = self.current_batch.Pos2[:step]
        input_batch = (input_words, input_pos1s, input_pos2s)
        input_lengths = self.current_batch.Text[1]
        assert input_lengths[0].item() == step
        with torch.no_grad():
            semantic_emb = self.net.get_semantic_representation(input_batch, input_lengths)

        return semantic_emb

    def get_input_embedding_at(self, step):
        """
        Calculate input embedding for original word at this step

        Args:
            step: an integer to indicate current step index

        Returns:
            input embedding for this step index
        """
        input_word = self.raw_example_batch.Text[0][step]  # size([1])
        input_pos1 = self.raw_example_batch.Pos1[step]  # size([1])
        input_pos2 = self.raw_example_batch.Pos2[step]  # size([1])
        with torch.no_grad():
            step_emb = self.net.get_input_embedding((input_word, input_pos1, input_pos2))

        return step_emb

    def update_forward_lstm_state(self, step):
        input_words = self.current_batch.Text[0][step]  # 1 x word_emb_size
        input_pos1s = self.current_batch.Pos1[step]  # 1 x pos_emb_size
        input_pos2s = self.current_batch.Pos2[step]
        input_step_batch = (input_words, input_pos1s, input_pos2s)
        with torch.no_grad():
            self.forward_lstm_state = self.net.forward_lstm_unroll_step(input_step_batch, self.forward_lstm_state)

            # def set_reward_eta(self, reward_eta, change_name_flag=True):
            #     self.reward_eta = reward_eta
            #     if change_name_flag:
            #         key = 'policy_store_name_prefix'
            #         if '_eta' in self.config[key]:
            #             self.config[key] = self.config[key].split('_eta')[0]
            #         self.config[key] = self.config[key] + '_eta{}'.format(self.reward_eta)


def aggregate_loss(action_logps, rewards):
    losses = []
    for action_logp, reward in zip(action_logps, rewards):
        losses.append(action_logp * reward)
    final_loss = - torch.cat(losses).sum()  # minimize - log(prob(a|s)) * r
    return final_loss


def batch_aggregate_loss(seq_action_logps, mask_list, reward_list):
    assert len(mask_list) == len(reward_list)
    seq_len, batch_size, action_size = seq_action_logps.size()
    sample_cnt = len(mask_list)
    loss_mat = torch.zeros_like(seq_action_logps)  # Size([seq_len, batch_size, action_size])

    for seq_action_masks, seq_action_rewards in zip(mask_list, reward_list):
        loss_mat += seq_action_logps * seq_action_masks * seq_action_rewards
    final_loss = - torch.sum(loss_mat) / (batch_size * sample_cnt)

    return final_loss


ErasureDecision = namedtuple('ErasureDecision', ('example_id', 'words', 'raw_pred_logp',
                                                 'actions', 'new_pred_logp',
                                                 'erasure_ratio', 'reward'))


def discounted_rewards(raw_rewards, gamma=0.999):
    disc_rewards = []
    tmp_val = 0.0
    for r in raw_rewards[::-1]:
        tmp_val = r + gamma * tmp_val
        disc_rewards.append(tmp_val)
    disc_rewards.reverse()
    return disc_rewards


class EraseAgent(object):
    """
    An agent learns how to erase tokens to produce a rule
    """

    def __init__(self, config, state_size, device, resume_flag=True):
        print('-' * 15, 'EraseAgent Initialization', '-' * 15)
        self.config = dict(config)
        self.state_size = state_size
        self.device = device

        self.reward_gamma = self.config['policy_reward_gamma']
        self.policy_epsilon = self.config['policy_eps']
        self.policy_epsilon_decay = self.config['policy_eps_decay']

        # set policy network
        self.policy_net = ErasurePolicy(self.state_size, self.config['hidden_size'])
        # old version 2
        # self.policy_net = ErasurePolicy(self.state_size)
        if resume_flag:
            policy_cpt_path = os.path.join(self.config['model_dir'],
                                           self.config['policy_resume_name'])
            self.resume_policy_from(policy_cpt_path)
        self.policy_net.to(self.device)

        # set optimizer
        self.optimizer = self.get_optimizer(self.policy_net.parameters())

        self.episode_action_logps = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_action_probs = []

        print('Create erasure agent, reward_gamma={}, policy_eps={}, policy_eps_decay={}'.format(
            self.reward_gamma, self.policy_epsilon, self.policy_epsilon_decay))
        print('-' * 15, 'EraseAgent Initialization End', '-' * 15)

    def get_optimizer(self, parameters):
        opti_type = self.config['optimizer_type']
        p_lr = self.config['policy_lr']
        p_wd = self.config['policy_weight_decay']

        print('Choose {} optimizer (lr={}, weight_decay={})'.format(opti_type, p_lr, p_wd))
        if opti_type == 'SGD':
            optimizer = optim.SGD(parameters,
                                  lr=p_lr,
                                  weight_decay=p_wd)
        elif opti_type == 'Adagrad':
            optimizer = optim.Adagrad(parameters,
                                      lr=p_lr,
                                      weight_decay=p_wd)
        elif opti_type == 'Adadelta':
            optimizer = optim.Adadelta(parameters,
                                       lr=p_lr,
                                       weight_decay=p_wd)
        elif opti_type == 'Adam':
            optimizer = optim.Adam(parameters,
                                   lr=p_lr,
                                   weight_decay=p_wd)
        else:
            raise ValueError('Unsupported optimizer type:', opti_type)

        return optimizer

    def select_action(self, state, rand_flag=False, eps_flag=False, eps_value=1.0, train_flag=False):
        def get_reverse_prob(probs):
            # assume probs.size() is size([1, action_size])
            rev_idxs = torch.arange(probs.size(-1) - 1, -1, -1, device=probs.device).long()
            with torch.no_grad():
                rev_probs = torch.index_select(probs, -1, rev_idxs)
            return rev_probs

        if train_flag:
            self.policy_net.train()
            action_probs = self.policy_net(state)  # size([1, action_size])
        else:
            self.policy_net.eval()
            with torch.no_grad():
                action_probs = self.policy_net(state)  # size([1, action_size])
        action_logps = probs_to_logits(action_probs)  # size([1, action_size])

        if rand_flag:
            if eps_flag and random.random() < eps_value:
                # print('use epsilon random policy')
                action_rev_probs = get_reverse_prob(action_probs)
                m = dist.Categorical(probs=action_rev_probs)
            else:
                m = dist.Categorical(probs=action_probs)
            action = m.sample()  # size([1])
            # action_logp = m.log_prob(action)  # size([1])
        else:
            action = torch.argmax(action_probs, dim=-1)  # size([1])

        assert action.requires_grad is False
        action_logp = action_logps.gather(-1, action.unsqueeze(0)).squeeze(-1)  # size([1])
        action = action.item()
        self.episode_actions.append(action)
        self.episode_action_logps.append(action_logp)
        self.episode_action_probs.append(action_probs)

        return action

    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)

    def reset_episode_info(self):
        self.episode_action_logps = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_action_probs = []

    def batch_calcu_action_prob(self, state_emb_seq, state_seq_lens, train_flag=False):
        if train_flag:
            self.policy_net.train()  # enter train mode
            seq_action_probs = self.policy_net.get_batch_action_prob(state_emb_seq, state_seq_lens)
        else:
            self.policy_net.eval()  # enter eval mode
            with torch.no_grad():
                seq_action_probs = self.policy_net.get_batch_action_prob(state_emb_seq, state_seq_lens)
        # seq_action_probs.shape is Size([seq_len, batch_size, action_size])
        return seq_action_probs

    @staticmethod
    def batch_select_action(seq_action_probs, batch_seq_lens, rand_flag=False, eps_flag=False, eps_value=0.1):
        seq_batch_action_probs = seq_action_probs.detach().cpu().numpy()  # (seq_len, batch_size, action_size)
        seq_action_masks = torch.zeros_like(seq_action_probs)  # Size(seq_len, batch_size, action_size)
        max_seq_len, batch_size, action_size = seq_action_probs.size()
        for batch_idx, seq_len in enumerate(batch_seq_lens):
            for seq_idx in range(seq_len):
                action_prob = seq_batch_action_probs[seq_idx, batch_idx]
                if rand_flag:
                    if eps_flag and random.random() < eps_value:
                        # this hopes to prevent overfitting at training stage
                        action_idx = np.random.choice(action_size)
                        # action_idx = np.random.choice(action_size, p=action_prob[::-1])
                    else:
                        action_idx = np.random.choice(action_size, p=action_prob)
                else:
                    # this is the greedy strategy often used at inference stage
                    action_idx = np.argmax(action_prob)
                seq_action_masks[seq_idx, batch_idx, action_idx] = 1
        return seq_action_masks

    def batch_interact_with(self, env, sample_cnt, fix_example=None, train_flag=False,
                            rand_flag=False, eps_flag=False, eps_value=0.1):
        # get next example batch
        env.next_example(example=fix_example)

        # Size(seq_len, batch_size, action_size)
        seq_action_probs = self.batch_calcu_action_prob(env.input_emb_seq, env.input_seq_lens, train_flag=train_flag)
        seq_action_logps = probs_to_logits(seq_action_probs)

        batch_seq_lens = env.input_seq_lens.tolist()
        batch_size = len(batch_seq_lens)
        batch_mean_rewards = torch.zeros_like(env.input_seq_lens).float()  # Size(batch_size)

        mask_list = []
        reward_list = []
        for sample_idx in range(sample_cnt):
            # reset episode variables of env and agent
            env.reset_state()
            self.reset_episode_info()

            # Size([seq_len, batch_size, action_size]), dtype=torch.float
            seq_action_masks = self.batch_select_action(seq_action_probs, batch_seq_lens, rand_flag=rand_flag,
                                                        eps_flag=eps_flag, eps_value=eps_value)

            batch_rewards = env.batch_transition_with(seq_action_masks)  # Size([batch_size])
            batch_mean_rewards += batch_rewards  # Size([batch_size])
            # TODO: whether to add reward decay strategy
            seq_action_rewards = batch_rewards.unsqueeze(0).unsqueeze(-1).expand_as(seq_action_masks)
            # for torch 1.1.0
            seq_action_rewards = seq_action_rewards.contiguous()

            assert seq_action_masks.requires_grad is False
            assert seq_action_rewards.requires_grad is False
            mask_list.append(seq_action_masks)
            reward_list.append(seq_action_rewards)

        # reduce reward variance
        if sample_cnt > 1 or batch_size > 1:
            batch_mean_rewards /= sample_cnt  # Size([batch_size])
            base_mean_rewards = batch_mean_rewards.unsqueeze(0).unsqueeze(-1).expand_as(seq_action_probs)
            # cur_baseline = torch.sum(batch_mean_rewards) / batch_size  # Size([])
            for sample_idx in range(sample_cnt):
                # reward_list[sample_idx] -= cur_baseline  # Size([batch_size])
                reward_list[sample_idx] -= base_mean_rewards  # Size([batch_size])

        return seq_action_probs, seq_action_logps, mask_list, reward_list, batch_mean_rewards

    def batch_train(self, env, policy_store_prefix=None, rand_flag=True, eps_flag=True, eps_value=None):
        # set environment data iter
        batch_size = self.config['policy_batch_size']
        env.set_data_iter('train', train_mode=True, batch_size=batch_size)

        max_epoch = self.config['policy_max_epoch']
        sample_cnt = self.config['policy_sample_cnt']
        print_reward_freq = self.config['print_reward_freq']
        if eps_value is None:
            eps_value = self.policy_epsilon
        eps_decay = self.policy_epsilon_decay

        for epoch in range(max_epoch):
            print('[ Epoch {} starts ]'.format(epoch + 1))

            batch_mean_rewards_list = []
            for batch_index in range(len(env.env_data_set) // batch_size):
                # interact with environment with batch_size different examples
                # for each example, run sample_cnt episodes
                # collect action_logps and rewards for backward
                seq_action_probs, seq_action_logps, mask_list, reward_list, batch_mean_rewards = \
                    self.batch_interact_with(env, sample_cnt, train_flag=True,
                                             rand_flag=rand_flag, eps_flag=eps_flag, eps_value=eps_value)
                # get aggregated loss
                loss = batch_aggregate_loss(seq_action_logps, mask_list, reward_list)

                # backward and optimize
                self.policy_net.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_mean_rewards_list.append(batch_mean_rewards)
                if print_reward_freq > 0 and batch_index % print_reward_freq == 0:
                    prev_mean_rewards = torch.cat(batch_mean_rewards_list, dim=0)
                    batch_mean_rewards_list = [prev_mean_rewards]
                    cur_mean_reward = torch.mean(prev_mean_rewards).item()
                    num_examples = prev_mean_rewards.size(0)
                    print('{} Mini-batch {}, loss {} (mean episode reward {}, {} examples)'.format(
                        time.asctime(), batch_index, loss.item(), cur_mean_reward, num_examples))

            # self.dump_policy(epoch + 1, is_best_flag=True, policy_store_prefix=policy_store_prefix)

            total_mean_rewards = torch.cat(batch_mean_rewards_list, dim=0)
            final_mean_reward = torch.mean(total_mean_rewards).item()
            num_examples = total_mean_rewards.size(0)
            print('[ Epoch {} ends ] (mean episode reward {}, {} examples)'.format(
                epoch + 1, final_mean_reward, num_examples))

            if 0 < eps_decay < 1.0:
                eps_value *= eps_decay
                print('... adjust policy epsilon to {}'.format(eps_value))
        self.dump_policy(max_epoch, is_best_flag=True, policy_store_prefix=policy_store_prefix)

    def batch_eval(self, env, data_type='train', saved_pickle_file=None, rand_flag=False, eps_flag=False):
        # set environment data iter
        env.set_data_iter(data_type=data_type, train_mode=False, batch_size=1)
        print_freq = self.config['print_reward_freq']

        examples = env.env_data_set.examples
        print('{} Policy evaluation starts, {} examples in total'.format(time.asctime(), len(examples)))
        eid2example = dict([(int(ex.Id), ex) for ex in examples])
        ex_decisions = []

        for batch_index in range(len(examples)):
            if batch_index > 0 and batch_index % print_freq == 0:
                print('{} Having processed {} examples'.format(time.asctime(), batch_index))

            seq_action_probs, seq_action_logps, mask_list, reward_list, batch_mean_rewards = \
                self.batch_interact_with(env, 1, train_flag=False, rand_flag=rand_flag,
                                         eps_flag=eps_flag, eps_value=self.policy_epsilon)
            seq_action_masks = mask_list[0]  # Size([seq_len, batch_size, action_size])

            ex_id = env.raw_example_ids[0]
            ex_obj = eid2example.pop(ex_id, None)
            if ex_obj is None:
                print('Warning: some errors with example (id={}), ...skip...'.format(ex_id))
                continue
            ex_raw_pred_logp = env.raw_pred_logps[0]
            ex_actions = seq_action_masks[:, :, 1].squeeze().tolist()
            # ex_action_probs = seq_action_probs.squeeze().tolist()
            ex_new_pred_logp = env.current_pred_logps[0]
            ex_erasure_ratio = np.mean(ex_actions)
            ex_reward = batch_mean_rewards[0].item()
            # ex_decisions.append(ErasureDecision(ex_obj, ex_raw_pred_logp,
            ex_decisions.append(ErasureDecision(ex_id, ex_obj.Text, ex_raw_pred_logp,
                                                ex_actions, ex_new_pred_logp,
                                                ex_erasure_ratio, ex_reward))
        if len(eid2example) != 0:  # not all examples has been visited
            print('Warning: {} examples have not been visited'.format(len(eid2example)))

        if saved_pickle_file is not None:
            save_fp = os.path.join(self.config['model_dir'],
                                   saved_pickle_file)
            with open(save_fp, 'wb') as fout:
                pickle.dump(ex_decisions, fout)

        return ex_decisions

    def interact_with(self, env, batch_size, sample_cnt, fix_example=None,
                      rand_flag=False, eps_flag=False, eps_value=1.0, train_flag=False):
        total_action_logps = []
        total_rewards = []
        mean_episode_rewards = []

        for example_index in range(batch_size):
            env.next_example(example=fix_example)  # change example

            state_emb_seq = env.input_emb_seq
            tmp_action_logps = []
            tmp_rewards = []

            if train_flag:
                self.policy_net.pre_compute_attention_base(state_emb_seq)
            else:
                with torch.no_grad():
                    self.policy_net.pre_compute_attention_base(state_emb_seq)

            for sample_index in range(sample_cnt):
                # reset environment state and episode collections
                state = env.reset_state()
                self.reset_episode_info()
                done = False

                # start (s, a) -r-> s' transition
                for step_index in range(env.step_length):
                    action = self.select_action(state,
                                                rand_flag=rand_flag,
                                                eps_flag=eps_flag,
                                                eps_value=eps_value,
                                                train_flag=train_flag)
                    reward, state, done = env.transition_with(action)
                    self.add_episode_reward(reward)
                # print('Example {} sample {} reward {}'.format(example_index, sample_index, self.episode_rewards[-1]))
                assert done is True

                # collect log prob tensor (requires_grad = True) and discounted reward
                tmp_action_logps += self.episode_action_logps
                tmp_rewards += discounted_rewards(self.episode_rewards, gamma=self.reward_gamma)

                mean_episode_rewards.append(self.episode_rewards[-1])

            # reduce variance of rewards for current example
            if sample_cnt > 1:
                mean_reward = np.mean(tmp_rewards)
                tmp_rewards = [r - mean_reward for r in tmp_rewards]

            # collect total action_logps and rewards
            total_action_logps += tmp_action_logps
            total_rewards += tmp_rewards
        assert len(total_action_logps) == len(total_rewards)

        return total_action_logps, total_rewards, mean_episode_rewards

    def train(self, env, max_epoch=None, batch_size=None, sample_cnt=None,
              print_reward_freq=None, policy_store_prefix=None, rand_flag=True,
              eps_flag=True, eps_value=None, eps_decay=None):
        if max_epoch is None:
            max_epoch = self.config['policy_max_epoch']
        if batch_size is None:
            batch_size = self.config['policy_batch_size']
        if sample_cnt is None:
            sample_cnt = self.config['policy_sample_cnt']
        if print_reward_freq is None:
            print_reward_freq = self.config['print_reward_freq']
        if eps_value is None:
            eps_value = self.policy_epsilon
        if eps_decay is None:
            eps_decay = self.policy_epsilon_decay

        env.set_data_iter('train', train_mode=True, batch_size=batch_size)
        for epoch in range(max_epoch):
            print('[ Epoch', epoch + 1, 'starts ]')

            epoch_mean_epi_rewards = []
            for batch_index in range(len(env.env_data_set) // batch_size):
                # interact with environment with batch_size different examples
                # for each example, run sample_cnt episodes
                # collect action_logps and rewards for backward
                total_action_logps, total_rewards, mean_epi_rewards = self.interact_with(
                    env, batch_size, sample_cnt, rand_flag=rand_flag,
                    eps_flag=eps_flag, eps_value=eps_value, train_flag=True)

                # get aggregated loss
                loss = aggregate_loss(total_action_logps, total_rewards)

                # backward and optimize
                self.policy_net.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_mean_epi_rewards += mean_epi_rewards
                if print_reward_freq > 0 and batch_index % print_reward_freq == 0:
                    print('{} Mini-batch {}, loss {}, mean episode reward {}'.format(
                        time.asctime(), batch_index, loss.item(), np.mean(epoch_mean_epi_rewards)))

            self.dump_policy(epoch + 1, is_best_flag=True, policy_store_prefix=policy_store_prefix)
            print('[ Epoch {} ends ] (mean episode reward {})'.format(
                epoch + 1, np.mean(epoch_mean_epi_rewards)))

            if 0 < eps_decay < 1.0:
                eps_value *= eps_decay
                print('New policy epsilon: {}'.format(eps_value))

    def eval(self, env, saved_pickle_file=None, print_freq=1000, rand_flag=False, eps_flag=False):
        examples = env.env_data_set.examples
        print('{} Policy evaluation starts, {} examples in total'.format(time.asctime(), len(examples)))
        eid2example = dict([(int(ex.Id), ex) for ex in examples])
        ex_decisions = []

        for batch_index in range(len(examples)):
            if batch_index > 0 and batch_index % print_freq == 0:
                print('{} Having processed {} examples'.format(time.asctime(), batch_index))

            _ = self.interact_with(env, 1, 1,
                                   rand_flag=rand_flag,
                                   eps_flag=eps_flag, eps_value=self.policy_epsilon,
                                   train_flag=False)
            ex_id = env.raw_example_ids[0]
            ex_obj = eid2example.pop(ex_id, None)
            if ex_obj is None:
                print('Some errors with example (id={}), ...skip...'.format(ex_id))
                continue
            ex_raw_pred_logp = env.raw_pred_logps[0]
            ex_actions = self.episode_actions
            # with torch.no_grad():
            #     ex_action_probs = torch.cat(self.episode_action_probs, dim=0).tolist()
            ex_new_pred_logp = env.current_pred_logps[0]
            ex_erasure_ratio = np.mean(ex_actions)
            ex_reward = self.episode_rewards[-1]
            # ex_decisions.append(ErasureDecision(ex_obj, ex_raw_pred_logp,
            #                                     ex_actions, ex_action_probs,
            #                                     ex_new_pred_logp, ex_erasure_ratio, ex_reward))
            ex_decisions.append(ErasureDecision(int(ex_obj.Id), ex_obj.Text, ex_raw_pred_logp,
                                                ex_actions, ex_new_pred_logp,
                                                ex_erasure_ratio, ex_reward))
        if len(eid2example) != 0:  # not all examples has been visited
            print('{} examples have not been visited'.format(len(eid2example)))

        if saved_pickle_file is not None:
            save_fp = os.path.join(self.config['model_dir'],
                                   saved_pickle_file)
            with open(save_fp, 'wb') as fout:
                pickle.dump(ex_decisions, fout)

        return ex_decisions

    def dump_policy(self, epoch, is_best_flag=False, policy_store_prefix=None):
        if policy_store_prefix is None:
            fp_prefix = os.path.join(self.config['model_dir'],
                                     self.config['policy_store_name_prefix'])
        else:
            fp_prefix = os.path.join(self.config['model_dir'], policy_store_prefix)

        fn_suffix = '.{}'.format(epoch)
        policy_state = {
            'epoch': epoch,
            'policy': self.policy_net.state_dict()
        }
        save_checkpoint(policy_state, is_best_flag, fp_prefix, fn_suffix)
        print('*' * 5, 'Dump policy into', fp_prefix + fn_suffix)

    def resume_policy_from(self, policy_cpt_path):
        print('Resume policy from {}'.format(policy_cpt_path))
        resume_checkpoint(self.policy_net, policy_cpt_path,
                          strict=True, resume_key='policy', print_keys=('epoch',))


def aggregate_rule_info(rel_env, train_example_decisions,
                        num_label_proc=10, min_source_num=3, saved_pickle_file=None):
    # rule_str2train_ex_decs = get_decision_rules(train_example_decisions)
    # rule_strs = rule_str2train_ex_decs.keys()
    # get fine grained rules, drop rules that min source num < 3
    rule_train_ex_decs_list = merge_common_rules(train_example_decisions,
                                                 min_source_num=min_source_num,
                                                 short_pad_range=(1, 3),
                                                 mid_pad_range=(4, 9))
    rule_strs = [rule_str for rule_str, _ in rule_train_ex_decs_list]

    if len(rule_strs) == 0:
        print('Warning: no rules extracted')
        rule_infos = []
    else:
        label_funcs = [create_label_func(rule_str, 1) for rule_str in rule_strs]
        label_factory = LabelFactory(label_funcs)

        lf_train_coo_mat = label_factory.batch_annotate(rel_env.raw_train_examples,
                                                        num_func_per_task=5000,
                                                        num_label_proc=num_label_proc)
        lf_train_csc_mat = lf_train_coo_mat.tocsc()
        lf_dev_coo_mat = label_factory.batch_annotate(rel_env.raw_dev_examples,
                                                      num_label_proc=num_label_proc)
        lf_dev_csc_mat = lf_dev_coo_mat.tocsc()
        lf_test_coo_mat = label_factory.batch_annotate(rel_env.raw_test_examples,
                                                       num_label_proc=num_label_proc)
        lf_test_csc_mat = lf_test_coo_mat.tocsc()

        rule_infos = []
        for rule_idx, (rule_str, train_ex_decs) in enumerate(rule_train_ex_decs_list):
            # rule_train_ex_decs = rule_str2train_ex_decs[rule_str]
            rule_train_dec_eids = [ex_dec.example_id for ex_dec in train_ex_decs]
            mean_dec_pred_prob = np.mean([math.exp(ex_dec.new_pred_logp[1]) for ex_dec in train_ex_decs])

            rule_train_ex_rids = lf_train_csc_mat[:, rule_idx].indices.tolist()
            rule_train_example_ids = [int(rel_env.raw_train_examples[rid].Id) for rid in rule_train_ex_rids]
            rule_train_match_acc, _ = get_rule_quality(rule_train_ex_rids, rel_env.raw_train_examples)

            rule_dev_ex_rids = lf_dev_csc_mat[:, rule_idx].indices.tolist()
            rule_dev_example_ids = [int(rel_env.raw_dev_examples[rid].Id) for rid in rule_dev_ex_rids]
            rule_dev_match_acc, _ = get_rule_quality(rule_dev_ex_rids, rel_env.raw_dev_examples)

            rule_test_ex_rids = lf_test_csc_mat[:, rule_idx].indices.tolist()
            rule_test_example_ids = [int(rel_env.raw_test_examples[rid].Id) for rid in rule_test_ex_rids]
            rule_test_match_acc, _ = get_rule_quality(rule_test_ex_rids, rel_env.raw_test_examples)

            rule_infos.append((rule_str,
                               rule_train_dec_eids, mean_dec_pred_prob,
                               rule_train_example_ids, rule_train_match_acc,
                               rule_dev_example_ids, rule_dev_match_acc,
                               rule_test_example_ids, rule_test_match_acc))

        rule_infos.sort(key=lambda x: (len(x[5]), x[6], len(x[3])), reverse=True)

    if saved_pickle_file is not None and len(rule_infos) > 0:
        save_fp = os.path.join(rel_env.config['model_dir'],
                               saved_pickle_file)
        print('Dump aggregated rule information into {}'.format(save_fp))
        with open(save_fp, 'wb') as fout:
            pickle.dump(rule_infos, fout)

    return rule_infos


def update_rule_info(rel_env, prev_rule_infos, num_label_proc=10, saved_pickle_file=None,
                     train_flag=True, dev_flag=True, test_flag=True):
    if len(prev_rule_infos) == 0:
        print('Warning: no rules extracted')
        rule_infos = []
    else:
        prev_rule_strs = [ri[0] for ri in prev_rule_infos]
        label_funcs = [create_label_func(rule_str, 1) for rule_str in prev_rule_strs]
        label_factory = LabelFactory(label_funcs)

        if train_flag:
            lf_train_coo_mat = label_factory.batch_annotate(rel_env.raw_train_examples,
                                                            num_func_per_task=5000,
                                                            num_label_proc=num_label_proc)
            lf_train_csc_mat = lf_train_coo_mat.tocsc()
        else:
            lf_train_csc_mat = None

        if dev_flag:
            lf_dev_coo_mat = label_factory.batch_annotate(rel_env.raw_dev_examples,
                                                          num_label_proc=num_label_proc)
            lf_dev_csc_mat = lf_dev_coo_mat.tocsc()
        else:
            lf_dev_csc_mat = None

        if test_flag:
            lf_test_coo_mat = label_factory.batch_annotate(rel_env.raw_test_examples,
                                                           num_label_proc=num_label_proc)
            lf_test_csc_mat = lf_test_coo_mat.tocsc()
        else:
            lf_test_csc_mat = None

        rule_infos = []
        for rule_idx, prev_rule_info in enumerate(prev_rule_infos):
            rule_str = prev_rule_info[0]
            rule_train_dec_eids = prev_rule_info[1]
            mean_dec_pred_prob = prev_rule_info[2]

            if train_flag:
                rule_train_ex_rids = lf_train_csc_mat[:, rule_idx].indices.tolist()
                rule_train_example_ids = [int(rel_env.raw_train_examples[rid].Id) for rid in rule_train_ex_rids]
                rule_train_match_acc, _ = get_rule_quality(rule_train_ex_rids, rel_env.raw_train_examples)
            else:
                rule_train_example_ids = prev_rule_info[3]
                rule_train_match_acc = prev_rule_info[4]

            if dev_flag:
                rule_dev_ex_rids = lf_dev_csc_mat[:, rule_idx].indices.tolist()
                rule_dev_example_ids = [int(rel_env.raw_dev_examples[rid].Id) for rid in rule_dev_ex_rids]
                rule_dev_match_acc, _ = get_rule_quality(rule_dev_ex_rids, rel_env.raw_dev_examples)
            else:
                rule_dev_example_ids = prev_rule_info[5]
                rule_dev_match_acc = prev_rule_info[6]

            if test_flag:
                rule_test_ex_rids = lf_test_csc_mat[:, rule_idx].indices.tolist()
                rule_test_example_ids = [int(rel_env.raw_test_examples[rid].Id) for rid in rule_test_ex_rids]
                rule_test_match_acc, _ = get_rule_quality(rule_test_ex_rids, rel_env.raw_test_examples)
            else:
                rule_test_example_ids = prev_rule_info[7]
                rule_test_match_acc = prev_rule_info[8]

            rule_infos.append((rule_str,
                               rule_train_dec_eids, mean_dec_pred_prob,
                               rule_train_example_ids, rule_train_match_acc,
                               rule_dev_example_ids, rule_dev_match_acc,
                               rule_test_example_ids, rule_test_match_acc))

        rule_infos.sort(key=lambda x: (len(x[1]), len(x[3]), len(x[5])), reverse=True)

    if saved_pickle_file is not None and len(rule_infos) > 0:
        save_fp = os.path.join(rel_env.config['model_dir'],
                               saved_pickle_file)
        print('Dump aggregated rule information into {}'.format(save_fp))
        with open(save_fp, 'wb') as fout:
            pickle.dump(rule_infos, fout)

    return rule_infos
