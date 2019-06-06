# -*- coding: utf-8 -*-
# @Time    : 25/7/18 23:17
# @Author  : Shun Zheng

import os
import time
import subprocess as subproc


# relation specific directories
NYT_REL_DIRS = [
    'nyt_bus_company',
    'nyt_loc_administrative_divisions',
    'nyt_loc_capital',
    'nyt_loc_contains',
    'nyt_loc_country',
    'nyt_loc_neighborhood_of',
    'nyt_peo_nationality',
    'nyt_peo_place_lived',
    'nyt_peo_place_of_birth',
    'nyt_peo_place_of_death',
]
UW_REL_DIRS = [
    'uw_peo_per_origin',
    'uw_peo_place_lived',
    'uw_peo_place_of_birth',
    'uw_peo_place_of_death',
]
TOTAL_REL_DIRS = NYT_REL_DIRS + UW_REL_DIRS

# common global directory path
WORK_DIR = os.environ['WORK_DIR']
SRC_DIR_PREFIX = WORK_DIR
DATA_PREFIX = os.path.join(WORK_DIR, 'data')
MODEL_PREFIX = os.path.join(WORK_DIR, 'model')
CONFIG_PREFIX = os.path.join(WORK_DIR, 'configs')
LOG_PREFIX = os.path.join(WORK_DIR, 'logs')

REL_CONFIG_BASE = 'config.rel.base'
REL_CONFIG_NAME_TEMPLATE = 'config.rel.{}'
REL_TRAIN_COMMAND_TEMPLATE = './shell/train_rel.sh {} {}'

AGENT_CONFIG_BASE = 'config.agent.base'
AGENT_CONFIG_NAME_TEMPLATE = 'config.agent.{}'
AGENT_TRAIN_COMMAND_TEMPLATE = './shell/train_agent.sh {} {}'

DIAGNOSIS_CONFIG_BASE = 'config.diag.base'
DIAGNOSIS_CONFIG_NAME_TEMPALTE = 'config.diag.{}'
DIAGNOSIS_COMMAND_TEMPLATE = './shell/train_diag.sh {} {}'

TOTAL_EVAL_COMMAND_TEMPLATE = './shell/eval_total.sh {} {}'


def read_base_config(base_config_name):
    base_config_path = os.path.join(CONFIG_PREFIX, base_config_name)
    config_lines = []
    with open(base_config_path, 'r') as fin:
        for line in fin:
            config_lines.append(line.rstrip('\n'))

    return config_lines


def generate_new_config(rel_dir_name, config_base, config_name_template, **kwargs):
    print('{} Generate relation training config for {}'.format(time.asctime(), rel_dir_name))
    base_config_lines = read_base_config(config_base)
    new_config_name = config_name_template.format(rel_dir_name)
    new_config_path = os.path.join(CONFIG_PREFIX, new_config_name)
    new_config_lines = []
    for line in base_config_lines:
        # skip comments or blank line
        if line.startswith('#') or '=' not in line:
            new_config_lines.append(line)
            continue

        # modify core relation directory name
        if line.startswith('REL_DIR_NAME'):
            new_line = line.format(rel_dir_name)
            new_config_lines.append(new_line)
            continue

        # add other key word based arguments
        items = line.split('=')
        if len(items) != 2:
            new_config_lines.append(line)
            continue
        key, val = items
        if key in kwargs:
            val = kwargs[key]
        new_line = '{}={}'.format(key, val)
        new_config_lines.append(new_line)

    with open(new_config_path, 'w') as fout:
        for line in new_config_lines:
            fout.write(line + '\n')

    return new_config_path


def query_gpu_usage():
    gpu_paras = ['index', 'gpu_name', 'memory.used', 'memory.total']
    query_cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(gpu_paras))
    pipe = os.popen(query_cmd)
    gpu_id2mem_usage = {}
    for line in pipe.readlines():
        items = line.rstrip('\n').split(', ')
        gpu_id = items[0]
        mem_used = float(items[2].rstrip(' MiB'))
        mem_total = float(items[3].rstrip(' MiB'))
        mem_usage = mem_used / mem_total
        gpu_id2mem_usage[gpu_id] = mem_usage

    return gpu_id2mem_usage


def get_idle_gpu(max_mem_usage=0.01):
    gpu_id2mem_usage = query_gpu_usage()
    idle_gpus = []
    for gpu_id, mem_usage in gpu_id2mem_usage.items():
        if mem_usage <= max_mem_usage:
            idle_gpus.append((gpu_id, mem_usage))

    if len(idle_gpus) >= 1:
        # import random
        # idx = random.randint(0, len(idle_gpus)-1)
        idle_gpus.sort(key=lambda x: x[1])
        return idle_gpus[0][0]
    else:
        return None


def wait_for_idle_gpu(max_mem_usage=0.01, sleep_sec=1):
    print('{} Wait for idle gpu'.format(time.asctime()))
    gpu_id = get_idle_gpu(max_mem_usage)
    while gpu_id is None:
        time.sleep(sleep_sec)
        gpu_id = get_idle_gpu(max_mem_usage)
    print('{} Get idle gpu {}'.format(time.asctime(), gpu_id))
    return gpu_id


def batch_do_task(task_rel_args, base_config_file_name, config_file_name_tempalate, exec_command_template,
                  max_gpu_mem_usage=0.001):
    print('{} Batch task starts, {} tasks in total, command template {}'.format(
        time.asctime(), len(task_rel_args), exec_command_template))
    running_task_bags = []

    for task_rel, task_arg in task_rel_args:
        print('-'*50)
        print('[Relation] {} [Args] {}'.format(task_rel, task_arg))

        gpu_id = wait_for_idle_gpu(max_mem_usage=max_gpu_mem_usage, sleep_sec=1)
        new_config_path = generate_new_config(task_rel, base_config_file_name, config_file_name_tempalate, **task_arg)
        command = exec_command_template.format(new_config_path, gpu_id)

        print('{} Task starts {}'.format(time.asctime(), command))
        cur_env = os.environ.copy()
        proc = subproc.Popen(command, stdout=subproc.PIPE, shell=True, env=cur_env)

        running_task_bags.append((proc, task_rel, task_arg))

        # wait for current process apply enough gpu memory
        time.sleep(10)

    print('='*50)
    print('{} Wait for all processes to exit'.format(time.asctime()))

    wait_flag = True
    while wait_flag:
        unfinished_task_bags = []
        for proc, task_rel, task_arg in running_task_bags:
            if proc.poll() is None:
                unfinished_task_bags.append((proc, task_rel, task_arg))
        if len(unfinished_task_bags) == 0:
            wait_flag = False
        time.sleep(1)

    print('{} All processes exit, exit code {}'.format(time.asctime(), [tb[0].poll() for tb in running_task_bags]))

