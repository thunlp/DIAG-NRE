# -*- coding: utf-8 -*-
# @Time    : 5/3/18 15:26
# @Author  : Shun Zheng

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def detail_lstm_cell(input_step_batch, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input_step_batch, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

    in_gate = F.sigmoid(in_gate)
    forget_gate = F.sigmoid(forget_gate)
    cell_gate = torch.tanh(cell_gate)
    out_gate = F.sigmoid(out_gate)

    cy = (forget_gate * cx) + (in_gate * cell_gate)
    hy = out_gate * torch.tanh(cy)

    return hy, cy, in_gate, forget_gate, cell_gate, out_gate


class BinarySoftNLLLoss(nn.Module):
    def __init__(self, size_average=True):
        super(BinarySoftNLLLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input_logp, input_marginals):
        batch_size, class_size = input_logp.shape
        na_marginals = 1 - input_marginals
        target_prob = torch.cat([na_marginals.unsqueeze(-1), input_marginals.unsqueeze(-1)], dim=-1)
        total_loss = - torch.sum(input_logp * target_prob)
        if self.size_average:
            return total_loss / batch_size
        else:
            return total_loss


class WordLSTM(nn.Module):
    """
    Word LSTM model which implements the paper,
    'Automatic rule extraction from long short term memory network'.
    """

    def __init__(self, word_vocab_size, word_vec_size, hidden_size, class_size,
                 input_emb_size=None, pre_word_vecs=None, use_cuda=False):
        super(WordLSTM, self).__init__()
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.word_vocab_size = word_vocab_size
        self.class_size = class_size
        if input_emb_size is None:
            self.input_emb_size = word_vec_size
        else:
            self.input_emb_size = input_emb_size
        self.use_cuda = use_cuda

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_vec_size)
        if pre_word_vecs is not None:
            self.word_embedding.weight.data.copy_(pre_word_vecs)

        self.lstm = nn.LSTM(self.input_emb_size, self.hidden_size, num_layers=1, bidirectional=False)
        self.out_fc = nn.Linear(self.hidden_size, self.class_size, bias=False)

    def get_lstm_output(self, input_emb, input_lengths):
        # input_batch: seq_len x batch_size x input_emb_size, input_lengths: [seq_len_1, ..., seq_len_[batch_size]]
        # packed_input: num_non_pad_token x input_emb_size, [batch_size_1, ..., batch_size_[seq_len]]
        packed_input = pack_padded_sequence(input_emb, list(input_lengths))

        # packed_output: num_non_pad_token x hidden_size, [batch_size_1, ..., batch_size_[seq_len]]
        packed_output, (hs, cs) = self.lstm(packed_input, hx=None)

        # lstm_output: seq_len x batch_size x hidden_size, [seq_len_1, ..., seq_len_[batch_size]]
        lstm_output, _ = pad_packed_sequence(packed_output)

        return lstm_output, (hs, cs)

    def get_detail_lstm_output(self, input_emb):
        seq_len, batch_size, _ = input_emb.size()  # seq_len x batch_size x input_emb_size

        hx = Variable(input_emb.data.new(batch_size, self.hidden_size).zero_(), requires_grad=False)
        hx, cx = hx, hx

        w_ih = self.lstm.weight_ih_l0
        b_ih = self.lstm.bias_ih_l0
        w_hh = self.lstm.weight_hh_l0
        b_hh = self.lstm.bias_hh_l0

        h_states, c_states, i_gates, f_gates, c_gates, o_gates = [], [], [], [], [], []
        for i in range(seq_len):
            # hx, cx, ig, fg, cg, og: batch_size x hidden_size
            hx, cx, ig, fg, cg, og = detail_lstm_cell(input_emb[i], (hx, cx), w_ih, w_hh, b_ih, b_hh)
            h_states.append(hx.unsqueeze(0))
            c_states.append(cx.unsqueeze(0))
            i_gates.append(ig.unsqueeze(0))
            f_gates.append(fg.unsqueeze(0))
            c_gates.append(cg.unsqueeze(0))
            o_gates.append(og.unsqueeze(0))

        # seq_len x batch_size x hidden_size
        h_states = torch.cat(h_states, dim=0)
        c_states = torch.cat(c_states, dim=0)
        i_gates = torch.cat(i_gates, dim=0)
        f_gates = torch.cat(f_gates, dim=0)
        c_gates = torch.cat(c_gates, dim=0)
        o_gates = torch.cat(o_gates, dim=0)

        return h_states, c_states, i_gates, f_gates, c_gates, o_gates

    def get_input_embedding(self, input_batch):
        # for word lstm, get input embedding by just looking up word vectors
        input_emb = self.word_embedding(input_batch)

        return input_emb

    def forward(self, input_batch, input_lengths):
        # get input embedding
        input_emb = self.get_input_embedding(input_batch)  # seq_len x batch_size x input_emb_size
        # run lstm forward process
        lstm_output, (hs, cs) = self.get_lstm_output(input_emb, input_lengths)
        # get last step lstm output
        # 1. using hidden state
        batch_last_hidden = hs.squeeze(0)
        # 2. using gather
        # batch_last_hidden = self.get_last_hidden_by_gather(lstm_output, input_lengths)
        # 3. using advanced indexing
        # batch_last_hidden = self.get_last_hidden_by_indexing(lstm_output, input_lengths)

        batch_logits = self.out_fc(batch_last_hidden)

        return F.log_softmax(batch_logits, dim=-1)

    def decompose(self, input_batch, input_lengths, unmask_default_value=1):
        # get input embedding
        input_emb = self.get_input_embedding(input_batch)  # seq_len x batch_size x input_emb_size
        seq_len, batch_size, _ = input_emb.size()

        # seq_len x batch_size x hidden_size
        hxs, cxs, igs, fgs, cgs, ogs = self.get_detail_lstm_output(input_emb)

        # using advanced indexing to get the last output gate value
        last_out_gate = self.get_last_hidden_units(ogs, input_lengths)  # batch_size x hidden_size
        last_out_gate = last_out_gate.unsqueeze(0).expand(seq_len, batch_size, self.hidden_size)

        # create mask, seq_len x batch_size
        length_mask = self.create_length_mask(input_lengths, seq_len, batch_size)

        # get cell differences
        tanh_cxs = torch.tanh(cxs)
        cell_diffs = [tanh_cxs[0].unsqueeze(0)]
        for i in range(1, seq_len):
            cell_diffs.append((tanh_cxs[i] - tanh_cxs[i-1]).unsqueeze(0))
        cell_diffs = torch.cat(cell_diffs, dim=0)  # seq_len x batch_size x hidden_size

        out_diffs = last_out_gate * cell_diffs  # seq_len x batch_size x hidden_size
        out_logits = F.linear(out_diffs, self.out_fc.weight)  # seq_len x batch_size x class_size
        out_beta = torch.exp(out_logits)  # seq_len x batch_size x class_size
        out_mask = length_mask.unsqueeze(2).expand(seq_len, batch_size, self.class_size)
        out_beta[out_mask == 0] = unmask_default_value

        return out_beta, out_mask

    def additive_decompose(self, input_batch, input_lengths, unmask_default_value=1):
        # get input embedding
        input_emb = self.get_input_embedding(input_batch)  # seq_len x batch_size x input_emb_size
        seq_len, batch_size, _ = input_emb.size()
        # seq_len x batch_size x hidden_size
        hxs, cxs, igs, fgs, cgs, ogs = self.get_detail_lstm_output(input_emb)

        # using advanced indexing to get the last output gate value
        last_out_gate = self.get_last_hidden_units(ogs, input_lengths)
        last_out_gate = last_out_gate.unsqueeze(0).expand(seq_len, batch_size, self.hidden_size)

        # create mask, seq_len x batch_size
        length_mask = self.create_length_mask(input_lengths, seq_len, batch_size)

        # ByteTensor, seq_len x batch_size x hidden_size
        hidden_mask = length_mask.unsqueeze(2).expand(seq_len, batch_size, self.hidden_size)

        fgs[hidden_mask == 0] = 1.0
        cxs[hidden_mask == 0] = 0.0

        for i in range(seq_len-2, 0, -1):
            fgs[i, :, :] = fgs[i, :, :] * fgs[i+1, :, :]
            cxs[i, :, :] = cxs[i, :, :] * fgs[i+1, :, :]

        tanh_cxs = torch.tanh(cxs)
        cell_diffs = [tanh_cxs[0].unsqueeze(0)]
        for i in range(1, seq_len):
            cell_diffs.append((tanh_cxs[i] - tanh_cxs[i-1]).unsqueeze(0))
        cell_diffs = torch.cat(cell_diffs, dim=0)

        out_diffs = last_out_gate * cell_diffs  # seq_len x batch_size x hidden_size
        out_logits = F.linear(out_diffs, self.out_fc.weight)  # seq_len x batch_size x class_size
        out_gamma = torch.exp(out_logits)  # seq_len x batch_size x class_size
        out_mask = length_mask.unsqueeze(2).expand(seq_len, batch_size, self.class_size)
        out_gamma[out_mask == 0] = unmask_default_value

        return out_gamma, out_mask

    def get_last_hidden_units(self, batch_hiddens, input_lengths):
        # using advanced indexing to get the last hidden units of each batch
        batch_size = batch_hiddens.size(1)
        seq_len_idxs = input_lengths - 1
        batch_idxs = torch.arange(batch_size).long()
        if self.use_cuda:
            batch_idxs = batch_idxs.cuda()
        # batch_size x hidden_size
        last_hiddens = batch_hiddens[seq_len_idxs, batch_idxs, :]

        # extract data by using gather operation
        # seq_end_idx_mat = seq_len_idxs.view(1, -1, 1).expand(1, -1, self.hidden_size)  # 1 x batch_size x hidden_size
        # sliced_lstm_output = torch.gather(lstm_output, 0, seq_end_idx_mat)  # 1 x batch_size x hidden_size
        # batch_last_hidden = sliced_lstm_output.squeeze(0)  # batch_size x hidden_size

        return last_hiddens

    def create_length_mask(self, input_lengths, seq_len, batch_size):
        seq_len_mat = torch.arange(seq_len).unsqueeze(1).expand(seq_len, batch_size).long()
        if self.use_cuda:
            seq_len_mat = seq_len_mat.cuda()
        in_len_mat = input_lengths.unsqueeze(0).expand(seq_len, batch_size).long()
        # ByteTensor, seq_len x batch_size
        length_mask = Variable(seq_len_mat < in_len_mat, requires_grad=False)

        return length_mask


class RelationLSTM(WordLSTM):
    """
    An advanced version of the word LSTM model that contains position embeddings.
    """
    def __init__(self, word_vocab_size, word_vec_size, pos_vocab_size, pos_vec_size, hidden_size, class_size,
                 pre_word_vecs=None, use_cuda=False):
        input_emb_size = word_vec_size + 2 * pos_vec_size
        super(RelationLSTM, self).__init__(word_vocab_size, word_vec_size, hidden_size, class_size,
                                           input_emb_size=input_emb_size,
                                           pre_word_vecs=pre_word_vecs,
                                           use_cuda=use_cuda)

        self.pos_vocab_size = pos_vocab_size
        self.pos_vec_size = pos_vec_size
        self.pos1_embedding = nn.Embedding(self.pos_vocab_size, self.pos_vec_size)
        self.pos2_embedding = nn.Embedding(self.pos_vocab_size, self.pos_vec_size)

    def get_input_embedding(self, input_batch):
        input_word, input_pos1, input_pos2 = input_batch
        input_word_emb = self.word_embedding(input_word)  # seq_len x batch_size x word_vec_size
        input_pos1_emb = self.pos1_embedding(input_pos1)  # seq_len x batch_size x pos_vec_size
        input_pos2_emb = self.pos2_embedding(input_pos2)  # seq_len x batch_size x pos_vec_size
        # input_emb: seq_len x batch_size x input_emb_size
        input_emb = torch.cat((input_word_emb, input_pos1_emb, input_pos2_emb), dim=-1)

        return input_emb


class RelationPCNN(nn.Module):
    """
    Implement model described in
    'Distant supervision for relation extraction via piece-wise convolutional neural networks'.
    """
    def __init__(self, word_vocab_size, word_vec_size, pos_vocab_size, pos_vec_size, class_size, ent_pos_id,
                 filter_size=230, window_size=3, pre_word_vecs=None, last_dropout_p=0.5):
        super(RelationPCNN, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.word_vec_size = word_vec_size
        self.pos_vocab_size = pos_vocab_size
        self.pos_vec_size = pos_vec_size
        self.input_emb_size = word_vec_size + 2 * pos_vec_size
        self.class_size = class_size
        self.ent_pos_id = ent_pos_id
        self.filter_size = filter_size
        self.window_size = window_size
        assert self.window_size % 2 != 0

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_vec_size)
        if pre_word_vecs is not None:
            self.word_embedding.weight.data.copy_(pre_word_vecs)
        self.pos1_embedding = nn.Embedding(self.pos_vocab_size, self.pos_vec_size)
        self.pos2_embedding = nn.Embedding(self.pos_vocab_size, self.pos_vec_size)

        self.conv = nn.Conv2d(1, self.filter_size,
                              kernel_size=(self.window_size, self.input_emb_size),
                              padding=(self.window_size//2, 0))
        self.out_fc = nn.Linear(self.filter_size*3, self.class_size, bias=True)
        self.last_dropout = nn.Dropout(p=last_dropout_p)

    def get_input_embedding(self, input_batch):
        input_word, input_pos1, input_pos2 = input_batch
        input_word_emb = self.word_embedding(input_word)  # seq_len x batch_size x word_vec_size
        input_pos1_emb = self.pos1_embedding(input_pos1)  # seq_len x batch_size x pos_vec_size
        input_pos2_emb = self.pos2_embedding(input_pos2)  # seq_len x batch_size x pos_vec_size
        # input_emb: seq_len x batch_size x input_emb_size
        input_emb = torch.cat((input_word_emb, input_pos1_emb, input_pos2_emb), dim=-1)

        return input_emb

    def get_pos_mask(self, input_batch, input_lengths):
        _, input_pos1, input_pos2 = input_batch
        seq_len = input_pos1.size(0)
        input_pos1 = input_pos1.transpose(0, 1)  # [batch_size, seq_len]
        input_pos2 = input_pos2.transpose(0, 1)  # [batch_size, seq_len]

        word_range_mask = torch.arange(seq_len, dtype=input_lengths.dtype, device=input_lengths.device)  # [seq_len]
        input_len_mat = input_lengths.unsqueeze(1).expand_as(input_pos1)  # [batch_size] -> [batch_size, seq_len]
        word_range_mask = word_range_mask.unsqueeze(0).expand_as(input_pos1) < input_len_mat  # [batch_size, seq_len]

        before_pos1_mask = (input_pos1 <= self.ent_pos_id) * word_range_mask
        after_pos1_mask = (input_pos1 > self.ent_pos_id) * word_range_mask
        before_pos2_mask = (input_pos2 <= self.ent_pos_id) * word_range_mask
        after_pos2_mask = (input_pos2 > self.ent_pos_id) * word_range_mask

        # [batch_size, seq_len]
        first_pos_mask = before_pos1_mask * before_pos2_mask
        mid_pos_mask = after_pos1_mask * before_pos2_mask + before_pos1_mask * after_pos2_mask
        last_pos_mask = after_pos1_mask * after_pos2_mask

        pos_masks = [first_pos_mask, mid_pos_mask, last_pos_mask]
        for idx, mask in enumerate(pos_masks):
            pos_masks[idx] = mask.float()

        return pos_masks

    def get_batch_logits(self, batch_emb, pos_masks):
        batch_conv = self.conv(batch_emb)  # [batch_size, filter_size, seq_len, 1]
        batch_conv = batch_conv.squeeze(-1)  # [batch_size, filter_size, seq_len]

        batch_poolings = []
        for pos_mask in pos_masks:
            pos_mask = pos_mask.unsqueeze(1).expand_as(batch_conv)
            batch_pos_pooling, _ = torch.max(pos_mask*batch_conv, dim=-1)  # [batch_size, filter_size]
            batch_poolings.append(batch_pos_pooling)

        batch_repre = torch.tanh(torch.cat(batch_poolings, dim=-1))  # [batch_size, filter_size*3]
        batch_repre = self.last_dropout(batch_repre)

        batch_logits = self.out_fc(batch_repre)  # [batch_size, class_size]

        return batch_logits

    def get_pred_probs(self, input_batch, input_lengths):
        input_emb = self.get_input_embedding(input_batch)  # [seq_len, batch_size, input_emb_size]
        pos_masks = self.get_pos_mask(input_batch, input_lengths)  # list of [batch_size, seq_len]
        batch_emb = input_emb.transpose(0, 1).unsqueeze(1)  # [batch_size, 1, seq_len, input_emb_size]
        batch_logits = self.get_batch_logits(batch_emb, pos_masks)

        return F.softmax(batch_logits, dim=-1)

    def forward(self, input_batch, input_lengths):
        input_emb = self.get_input_embedding(input_batch)  # [seq_len, batch_size, input_emb_size]
        pos_masks = self.get_pos_mask(input_batch, input_lengths)  # list of [batch_size, seq_len]
        batch_emb = input_emb.transpose(0, 1).unsqueeze(1)  # [batch_size, 1, seq_len, input_emb_size]
        batch_logits = self.get_batch_logits(batch_emb, pos_masks)

        return F.log_softmax(batch_logits, dim=-1)

    def get_init_state_info(self, input_batch, input_lengths):
        input_emb = self.get_input_embedding(input_batch)  # [seq_len, batch_size, input_emb_size]
        pos_masks = self.get_pos_mask(input_batch, input_lengths)  # list of [batch_size, seq_len]
        batch_emb = input_emb.transpose(0, 1).unsqueeze(1)  # [batch_size, 1, seq_len, input_emb_size]
        batch_logits = self.get_batch_logits(batch_emb, pos_masks)
        batch_pred_logp = F.log_softmax(batch_logits, dim=-1)

        return input_emb, batch_pred_logp


class RelationAttBiLSTM(nn.Module):
    """
    Implement model described in
    'Attention-based bidirectional long short-term memory networks for relation extraction'.
    """

    def __init__(self, word_vocab_size, word_vec_size, pos_vocab_size, pos_vec_size, hidden_size, class_size,
                 pre_word_vecs=None, emb_dropout_p=0.5, lstm_dropout_p=0.5, last_dropout_p=0.5):
        super(RelationAttBiLSTM, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.word_vec_size = word_vec_size
        self.pos_vocab_size = pos_vocab_size
        self.pos_vec_size = pos_vec_size
        self.input_emb_size = word_vec_size + 2 * pos_vec_size
        self.hidden_size = hidden_size
        self.class_size = class_size

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_vec_size)
        if pre_word_vecs is not None:
            self.word_embedding.weight.data.copy_(pre_word_vecs)
        self.pos1_embedding = nn.Embedding(self.pos_vocab_size, self.pos_vec_size)
        self.pos2_embedding = nn.Embedding(self.pos_vocab_size, self.pos_vec_size)

        # Note it is a bidirectional lstm layer
        self.lstm = nn.LSTM(self.input_emb_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.word_att = nn.Linear(self.hidden_size*2, 1, bias=False)
        self.out_fc = nn.Linear(self.hidden_size*2, self.class_size, bias=True)

        # add dropout layer for regularization
        self.emb_dropout = nn.Dropout(p=emb_dropout_p)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_p)
        self.last_dropout = nn.Dropout(p=last_dropout_p)

    def get_input_embedding(self, input_batch):
        input_word, input_pos1, input_pos2 = input_batch
        input_word_emb = self.word_embedding(input_word)  # seq_len x batch_size x word_vec_size
        input_pos1_emb = self.pos1_embedding(input_pos1)  # seq_len x batch_size x pos_vec_size
        input_pos2_emb = self.pos2_embedding(input_pos2)  # seq_len x batch_size x pos_vec_size
        # input_emb: seq_len x batch_size x input_emb_size
        input_emb = torch.cat((input_word_emb, input_pos1_emb, input_pos2_emb), dim=-1)

        return input_emb

    def get_lstm_output(self, input_emb, input_lengths):
        # input_batch: seq_len x batch_size x input_emb_size, input_lengths: [seq_len_1, ..., seq_len_[batch_size]]
        # packed_input: num_non_pad_token x input_emb_size, [batch_size_1, ..., batch_size_[seq_len]]
        packed_input = pack_padded_sequence(input_emb, list(input_lengths))

        # packed_output: num_non_pad_token x 2 hidden_size, [batch_size_1, ..., batch_size_[seq_len]]
        packed_output, (hs, cs) = self.lstm(packed_input, hx=None)

        # lstm_output: seq_len x batch_size x 2 hidden_size, [seq_len_1, ..., seq_len_[batch_size]]
        lstm_output, _ = pad_packed_sequence(packed_output)  # out of length range value will be padded 0

        return lstm_output, (hs, cs)

    def get_attentive_representation(self, lstm_output):
        m = torch.tanh(lstm_output)  # seq_len x batch_size x 2 hidden_size
        alpha = F.softmax(self.word_att(m), dim=0)  # seq_len x batch_size x 1
        alpha_mat = alpha.expand_as(lstm_output)  # seq_len x batch_size x 2 hidden_size
        att_mul_mat = alpha_mat * lstm_output
        att_average = att_mul_mat.sum(dim=0)  # batch_size x 2 hidden_size
        att_repre = torch.tanh(att_average)

        return att_repre

    def get_semantic_representation(self, input_batch, input_lengths):
        # get input embedding
        input_emb = self.get_input_embedding(input_batch)  # seq_len x batch_size x input_emb_size
        input_emb = self.emb_dropout(input_emb)
        # run bi-lstm forward process
        lstm_output, _ = self.get_lstm_output(input_emb, input_lengths)
        lstm_output = self.lstm_dropout(lstm_output)
        # get attention weighted representation
        att_repre = self.get_attentive_representation(lstm_output)  # batch_size x 2 hidden_size
        att_repre = self.last_dropout(att_repre)

        return att_repre

    def get_pred_probs(self, input_batch, input_lengths):
        att_repre = self.get_semantic_representation(input_batch, input_lengths)
        batch_logits = self.out_fc(att_repre)  # batch_size x class_size

        return F.softmax(batch_logits, dim=-1)

    def forward(self, input_batch, input_lengths):
        att_repre = self.get_semantic_representation(input_batch, input_lengths)
        batch_logits = self.out_fc(att_repre)  # batch_size x class_size

        return F.log_softmax(batch_logits, dim=-1)

    def get_init_state_info(self, input_batch, input_lengths):
        # get input embedding
        input_emb = self.get_input_embedding(input_batch)  # seq_len x batch_size x input_emb_size
        # run bi-lstm forward process
        lstm_output, _ = self.get_lstm_output(input_emb, input_lengths)  # seq_len x batch_size x 2*hidden_size
        # get attention weighted representation
        att_repre = self.get_attentive_representation(lstm_output)  # batch_size x 2 hidden_size
        att_repre = self.last_dropout(att_repre)
        # get out logits
        batch_logits = self.out_fc(att_repre)  # batch_size x class_size
        batch_pred_logp = F.log_softmax(batch_logits, dim=-1)  # batch_size x class_size

        return input_emb, batch_pred_logp

    def forward_lstm_unroll_step(self, input_step_batch, lstm_state_tuple):
        input_emb = self.get_input_embedding(input_step_batch)  # 1 x input_emb_size
        w_ih = self.lstm.weight_ih_l0
        b_ih = self.lstm.bias_ih_l0
        w_hh = self.lstm.weight_hh_l0
        b_hh = self.lstm.bias_hh_l0
        hx, cx, _, _, _, _ = detail_lstm_cell(input_emb, lstm_state_tuple,
                                              w_ih, w_hh, b_ih, b_hh)

        return hx, cx


class RelationSparseAttBiLSTM(RelationAttBiLSTM):
    """
    Add a mask layer to the attention-based bidirectional LSTM model in order to achieve sparsity.
    """

    def __init__(self, word_vocab_size, word_vec_size, pos_vocab_size, pos_vec_size, hidden_size, class_size, **kwargs):
        super(RelationSparseAttBiLSTM, self).__init__(word_vocab_size, word_vec_size,
                                                      pos_vocab_size, pos_vec_size,
                                                      hidden_size, class_size,
                                                      **kwargs)
        self.mask_embedding = nn.Embedding(word_vocab_size, 1)

    def init_mask_to_one(self):
        self.mask_embedding.weight.data.zero_().add_(1)  # initialize mask parameters to 1

    def only_train_mask_parameter(self):
        for pname, param in self.named_parameters():
            if pname == 'mask_embedding.weight':
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_parameters(self, mask_flag=False, non_mask_flag=False):
        for pname, param in self.named_parameters():
            if pname == 'mask_embedding.weight' and mask_flag:
                yield param
            if pname != 'mask_embedding.weight' and non_mask_flag:
                yield param

    def get_input_embedding(self, input_batch):
        input_emb = super(RelationSparseAttBiLSTM, self).get_input_embedding(input_batch)
        input_word = input_batch[0]
        input_mask = self.mask_embedding(input_word).expand_as(input_emb)
        input_emb = input_emb * input_mask
        return input_emb


class ErasurePolicy(nn.Module):
    def __init__(self, state_emb_size, hidden_size, action_size=2):
        print('Initialize context-aware parallel erasure policy')
        super(ErasurePolicy, self).__init__()
        self.state_emb_size = state_emb_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.bi_lstm = nn.LSTM(self.state_emb_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.out_fc = nn.Linear(self.state_emb_size + 2 * self.hidden_size, self.action_size)
        self.att_state_fc = nn.Linear(self.state_emb_size, self.hidden_size, bias=False)
        self.att_hidden_fc = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.att_v_fc = nn.Linear(self.hidden_size, 1, bias=False)
        # self.att_state_mat = nn.Parameter(torch.Tensor(self.hidden_size, self.state_emb_size))
        # self.att_hidden_mat = nn.Parameter(torch.Tensor(self.hidden_size, 2*self.hidden_size))
        # self.att_v = nn.Parameter(torch.Tensor(1, self.hidden_size))

        self.bi_hidden_emb_seq = None
        self.att_hidden_emb_seq = None

    def pre_compute_attention_base(self, state_emb_seq):
        # assume batch size is 1
        # state_emb_seq.shape = Size([seq_len, batch_size, state_emb_size])
        self.bi_hidden_emb_seq, _ = self.bi_lstm(state_emb_seq)  # Size(seq_len, batch_size, 2*hidden_size)
        # Size(seq_len, batch_size, hidden_size)
        self.att_hidden_emb_seq = self.att_hidden_fc(self.bi_hidden_emb_seq)

    def get_attentive_hidden_embedding(self, state_emb):
        # assume state_emb.Size is 1 x input_emb_size
        att_state_emb = self.att_state_fc(state_emb)  # 1 x hidden_size
        att_state_emb = att_state_emb.unsqueeze(0).expand_as(self.att_hidden_emb_seq)  # seq_len x 1 x hidden_size
        att_total_emb = torch.tanh(att_state_emb + self.att_hidden_emb_seq)  # seq_len x 1 x hidden_size
        att_logit = self.att_v_fc(att_total_emb)  # seq_len x 1 x 1
        att_alpha = F.softmax(att_logit, dim=0).expand_as(self.bi_hidden_emb_seq)  # seq_len x 1 x 2*hidden_size
        att_semantic_emb = torch.sum(att_alpha * self.bi_hidden_emb_seq, 0)  # 1 x 2*hidden_size

        return att_semantic_emb

    def forward(self, state_emb):
        # assume state_emb.Size is 1 x input_emb_size
        att_semantic_emb = self.get_attentive_hidden_embedding(state_emb)  # 1 x 2*hidden_size
        x = torch.cat([state_emb, att_semantic_emb], dim=-1)  # 1 x (state_emb_size + 2*hidden_size)
        out_logit = self.out_fc(x)  # 1 x action_size

        return F.softmax(out_logit, dim=-1)

    def get_batch_lstm_output(self, state_emb_seq, state_seq_lens):
        # state_emb_seq.shape is Size(seq_len, batch_size, state_emb_size)
        # input_lengths: [seq_len_1, ..., seq_len_[batch_size]]
        # packed_input: num_non_pad_token x input_emb_size, [batch_size_1, ..., batch_size_[seq_len]]
        packed_input = pack_padded_sequence(state_emb_seq, list(state_seq_lens))

        # packed_output: num_non_pad_token x 2 hidden_size, [batch_size_1, ..., batch_size_[seq_len]]
        packed_output, (hs, cs) = self.bi_lstm(packed_input, hx=None)

        # lstm_output: seq_len x batch_size x 2 hidden_size, [seq_len_1, ..., seq_len_[batch_size]]
        lstm_output, _ = pad_packed_sequence(packed_output)  # out of length range value will be padded 0

        return lstm_output, (hs, cs)

    def get_batch_context_embedding(self, state_emb_seq, state_seq_lens):
        # state_emb_seq.shape = Size([max_seq_len, batch_size, state_emb_size])
        max_seq_len = state_emb_seq.size(0)

        # pre-compute bi lstm output, Size([max_seq_len, batch_size, 2*hidden_size])
        # bi_lstm_seq, _ = self.bi_lstm(state_emb_seq)
        bi_lstm_seq, _ = self.get_batch_lstm_output(state_emb_seq, state_seq_lens)
        # expand to Size([max_seq_len, max_seq_len, batch_size, 2*hidden_size])
        bi_lstm_mat = bi_lstm_seq.unsqueeze(0).expand([max_seq_len, -1, -1, -1])

        # Size([max_seq_len, batch_size, hidden_size])
        att_state_seq = self.att_state_fc(state_emb_seq)
        att_hidden_seq = self.att_hidden_fc(bi_lstm_seq)

        # expand to Size([max_seq_len, max_seq_len, batch_size, hidden_size]) to compute attention matrix
        att_state_mat = att_state_seq.unsqueeze(1).expand([-1, max_seq_len, -1, -1])
        att_hidden_mat = att_hidden_seq.unsqueeze(0).expand([max_seq_len, -1, -1, -1])
        att_sum_mat = torch.tanh(att_state_mat + att_hidden_mat)

        # compute attention matrix
        att_logit_mat = self.att_v_fc(att_sum_mat)  # Size([max_seq_len, max_seq_len, batch_size, 1])
        # expand to bi_lstm_mat.Size()
        att_alpha_mat = F.softmax(att_logit_mat, dim=1).expand_as(bi_lstm_mat)
        att_avg_seq = torch.sum(att_alpha_mat*bi_lstm_mat, 1)  # Size([max_seq_len, batch_size, 2*hidden_size])

        return att_avg_seq

    def get_batch_action_prob(self, state_emb_seq, state_seq_lens):
        # assume state_emb_seq.shape is Size([seq_len, batch_size, state_emb_size])
        # get attentive context embedding, Size([seq_len, batch_size, 2*hidden_size])
        batch_context = self.get_batch_context_embedding(state_emb_seq, state_seq_lens)
        # shape = Size([seq_len, batch_size, state_emb_size + 2*hidden_size])
        batch_state_context = torch.cat([state_emb_seq, batch_context], dim=-1)
        batch_logit = self.out_fc(batch_state_context)  # Size([seq_len, batch_size, action_size])

        return F.softmax(batch_logit, dim=-1)

