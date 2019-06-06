# -*- coding: utf-8 -*-
# @Time    : 15/4/18 15:08
# @Author  : Shun Zheng

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import torch.optim as optim


class TruncateSGD(Optimizer):
    r""" Implements truncated stochastic gradient descent to achieve sparsity.

    The algorithm is based on the paper 'Sparse Online Learning via Truncated Gradient'.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        gravity (float): the gravity parameter (L1 penalty)
        truncate_freq (int, optional): the truncated frequency (default: 10)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=required, gravity=required, truncate_freq=1, weight_decay=0):
        defaults = dict(lr=lr, gravity=gravity, truncate_freq=truncate_freq, weight_decay=weight_decay)
        super(TruncateSGD, self).__init__(params, defaults)

        if not isinstance(truncate_freq, int) or truncate_freq <= 0:
            raise ValueError('truncate_freq should be integer and greater than 0',
                             'while type(truncate_freq) =', torch.typename(truncate_freq),
                             'truncate_freq =', truncate_freq)

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for pg in self.param_groups:
            lr = pg['lr']
            g = pg['gravity']
            K = pg['truncate_freq']
            weight_decay = pg['weight_decay']

            truncate_state = self.state['truncate_state']
            if 'index' not in truncate_state:
                truncate_state['index'] = 1
            truncate_index = truncate_state['index']

            for p in pg['params']:
                if p.grad is None:
                    continue

                # # =============== debug code start ===============
                # if 'global_step' not in self.state:
                #     self.state['global_step'] = 0
                # else:
                #     self.state['global_step'] += 1
                # if self.state['global_step'] % 20 == 0:
                #     print('-'*5 + 'Enter TruncateSGD step')
                #     print('index:{}, param.size:{}, lr:{}, g:{}, K:{}, weight_decay:{}'.format(
                #         truncate_index, p.size(), lr, g, K, weight_decay))
                #     num_nz_mask = len(p.data.nonzero())
                #     print('non-zero coefficient mask {}'.format(num_nz_mask))
                # # =============== debug code end ===============

                # gradient step
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)
                p.data.add_(-lr, p_grad)

                # truncate step
                if truncate_index > 0 and truncate_index % K == 0:
                    shrink_param = lr * K * g
                    p.data.copy_(F.softshrink(p.data, lambd=shrink_param).data)
                    truncate_state['index'] = 1
                else:
                    truncate_state['index'] += 1
        return loss


class TruncateAdam(optim.Adam):
    def __init__(self, params, lr_truncate=required, gravity=required, truncate_freq=1,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr_truncate=lr_truncate, gravity=gravity, truncate_freq=truncate_freq,
                        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(optim.Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = super(TruncateAdam, self).step(closure)

        for pg in self.param_groups:
            lr_truncate = pg['lr_truncate']
            g = pg['gravity']
            K = pg['truncate_freq']

            truncate_state = self.state['truncate_state']
            if 'index' not in truncate_state:
                truncate_state['index'] = 1
            truncate_index = truncate_state['index']

            for p in pg['params']:
                if p.grad is None:
                    continue

                # truncate step
                if truncate_index > 0 and truncate_index % K == 0:
                    shrink_param = lr_truncate * K * g
                    p.data.copy_(F.softshrink(p.data, lambd=shrink_param).data)
                    truncate_state['index'] = 1
                else:
                    truncate_state['index'] += 1

                # # =============== debug code start ===============
                # if 'global_step' not in self.state:
                #     self.state['global_step'] = 0
                # else:
                #     self.state['global_step'] += 1
                # if self.state['global_step'] % 50 == 0:
                #     print('-'*5 + 'Enter TruncateAdam step')
                #     print('index:{}, param.size:{}, lr:{}, g:{}, K:{}'.format(
                #         truncate_index, p.size(), lr_truncate, g, K))
                #     num_nz_mask = len(p.data.nonzero())
                #     print('non-zero coefficient mask {}'.format(num_nz_mask))
                # # =============== debug code end ===============

        return loss
