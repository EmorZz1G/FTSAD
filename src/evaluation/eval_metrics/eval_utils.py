import numpy as np
import random

# 定义生成连续段的函数，生成的信号长度为ts_len
# 异常片段
def generate_continuous_segments(num_points, num_segments, max_length, min_length=1):
    # signal = np.random.randint(0, 2, size=num_points)
    signal = np.zeros(num_points)
    for _ in range(num_segments):
        start = np.random.randint(0, len(signal) - max_length)
        end = start + max(np.random.randint(1, max_length + 1), min_length)
        signal[start:end] = 1
    return signal


def generate_random_scores(num_points, labels, typs=0, prob=0.1, init_seed=42):
    np.random.seed(init_seed)
    random.seed(init_seed)
    
    if typs == 0:
        # uniform noise
        anomaly_scores = np.random.rand(num_points)
    elif typs == 1:
        # gaussian noise
        anomaly_scores = np.random.randn(num_points)
        # min-max normalization
        anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    elif typs == 2:
        # the model has a probability of prob to predict an anomaly
        pred_prob = np.random.rand(num_points)
        # print(pred_prob)
        tmp = (pred_prob <= prob)
        # print(tmp)
        anomaly_scores = np.zeros(num_points)
        for i in range(0, num_points):
            if tmp[i] == 1:
                if labels[i] == 1:
                    anomaly_scores[i] = np.random.rand() * 0.1 + 0.9
                else:
                    anomaly_scores[i] = np.random.rand() * 0.05
            else:
                if labels[i] == 0:
                    anomaly_scores[i] = np.random.rand()* 0.1 + 0.9
                else:
                    anomaly_scores[i] = np.random.rand() * 0.05
    elif typs == 3:
        pass
                
    return anomaly_scores


configs = {
    '100k-20seg-50L': {'num_points': 100000, 'num_segments': 20, 'max_length': 60, 'min_length': 40}, # anomaly ratio: 0.01,
    '100k-200seg-50L': {'num_points': 100000, 'num_segments': 200, 'max_length': 60, 'min_length': 40}, # anomaly ratio: 0.1,
    '100k-20seg-50H': {'num_points': 100000, 'num_segments': 20, 'max_length': 99, 'min_length': 1}, # anomaly ratio: 0.01,
    '100k-200seg-50H': {'num_points': 100000, 'num_segments': 200, 'max_length': 99, 'min_length': 1}, # anomaly ratio: 0.1,
    '100k-50seg-20L': {'num_points': 100000, 'num_segments': 50, 'max_length': 30, 'min_length': 10}, # anomaly ratio: 0.01,
    '100k-500seg-20L': {'num_points': 100000, 'num_segments': 500, 'max_length': 30, 'min_length': 10}, # anomaly ratio: 0.1,
    '100k-50seg-20H': {'num_points': 100000, 'num_segments': 50, 'max_length': 39, 'min_length': 1}, # anomaly ratio: 0.01,
    '100k-500seg-20H': {'num_points': 100000, 'num_segments': 500, 'max_length': 39, 'min_length': 1}, # anomaly ratio: 0.1,
    '100k-10seg-100L': {'num_points': 100000, 'num_segments': 10, 'max_length': 110, 'min_length': 90}, # anomaly ratio: 0.01,
    '100k-100seg-100L': {'num_points': 100000, 'num_segments': 100, 'max_length': 110, 'min_length': 110}, # anomaly ratio: 0.1,
    '100k-10seg-100H': {'num_points': 100000, 'num_segments': 10, 'max_length': 199, 'min_length': 1}, # anomaly ratio: 0.01,
    '100k-100seg-100H': {'num_points': 100000, 'num_segments': 100, 'max_length': 199, 'min_length': 1}, # anomaly ratio: 0.1,
    '100k-2seg-500L': {'num_points': 100000, 'num_segments': 2, 'max_length': 550, 'min_length': 450}, # anomaly ratio: 0.01,
    '100k-20seg-500L': {'num_points': 100000, 'num_segments': 20, 'max_length': 550, 'min_length': 450}, # anomaly ratio: 0.1,
    '100k-2seg-500H': {'num_points': 100000, 'num_segments': 2, 'max_length': 999, 'min_length': 1}, # anomaly ratio: 0.01,
    '100k-20seg-500H': {'num_points': 100000, 'num_segments': 20, 'max_length': 999, 'min_length': 1}, # anomaly ratio: 0.1,

    '10k-2seg-50L': {'num_points': 10000, 'num_segments': 2, 'max_length': 60, 'min_length': 40}, # anomaly ratio: 0.01,
    '10k-20seg-50L': {'num_points': 10000, 'num_segments': 20, 'max_length': 60, 'min_length': 40}, # anomaly ratio: 0.1,
    '10k-2seg-50H': {'num_points': 10000, 'num_segments': 2, 'max_length': 99, 'min_length': 1}, # anomaly ratio: 0.01,
    '10k-20seg-50H': {'num_points': 10000, 'num_segments': 20, 'max_length': 99, 'min_length': 1}, # anomaly ratio: 0.1,
    '10k-5seg-20L': {'num_points': 10000, 'num_segments': 5, 'max_length': 30, 'min_length': 10}, # anomaly ratio: 0.01,
    '10k-50seg-20L': {'num_points': 10000, 'num_segments': 50, 'max_length': 30, 'min_length': 10}, # anomaly ratio: 0.1,
    '10k-5seg-20H': {'num_points': 10000, 'num_segments': 5, 'max_length': 39, 'min_length': 1}, # anomaly ratio: 0.01,
    '10k-50seg-20H': {'num_points': 10000, 'num_segments': 50, 'max_length': 39, 'min_length': 1}, # anomaly ratio: 0.1,
    '10k-1seg-100L': {'num_points': 10000, 'num_segments': 1, 'max_length': 110, 'min_length': 90}, # anomaly ratio: 0.01,
    '10k-10seg-100L': {'num_points': 10000, 'num_segments': 10, 'max_length': 110, 'min_length': 110}, # anomaly ratio: 0.1,
    '10k-1seg-100H': {'num_points': 10000, 'num_segments': 1, 'max_length': 199, 'min_length': 1}, # anomaly ratio: 0.01,
    '10k-10seg-100H': {'num_points': 10000, 'num_segments': 10, 'max_length': 199, 'min_length': 1}, # anomaly ratio: 0.1,
    '10k-2seg-500L': {'num_points': 10000, 'num_segments': 2, 'max_length': 550, 'min_length': 450}, # anomaly ratio: 0.1,
    '10k-2seg-500H': {'num_points': 10000, 'num_segments': 2, 'max_length': 999, 'min_length': 1}, # anomaly ratio: 0.1,
}

CASE_NUM = len(configs)

import os
from os.path import dirname as upd
file_pth = os.path.dirname(os.path.abspath(__file__))
file_pth = upd(upd(file_pth))
proj_pth = upd(file_pth)
data_pth = os.path.join(proj_pth, 'datasets')
import sys
sys.path.append(proj_pth)

from src.data_utils.SimAD_data_loader2 import get_loader_segment

def generate_dataset(case_idx=0, init_seed=42):
    def get_config(case_idx):
        if not isinstance(case_idx, int):
            if case_idx in configs.keys():
                return configs[case_idx]
            else:
                raise ValueError(f"If case_idx is not an integer, it must be a key in the config dictionary. Available keys: {list(config.keys())}")
        if case_idx < 0 or case_idx >= len(configs):
            raise ValueError(f"Invalid case index, case_idx={case_idx}. It should be between 0 and {len(config) - 1}.")
        else:
            return list(configs.values())[case_idx]
        

    config = get_config(case_idx)
    num_points = config['num_points']
    num_segments = config['num_segments']
    max_length = config['max_length']
    min_length = config['min_length']
    dataset_ver = list(configs.keys())[case_idx]
    # set random seed
    np.random.seed(init_seed)
    random.seed(init_seed)
    # generate anomaly segments
    gen_seg = generate_continuous_segments(num_points, num_segments, max_length, min_length)
    return dataset_ver, gen_seg


real_world_configs = [
    {'dataset_name': 'MSL'},
    {'dataset_name': 'NIPS_TS_Creditcard'},
    {'dataset_name': 'SWAT'},
    {'dataset_name': 'SMD_Ori_Pikled', 'index': "1-1"},
    {'dataset_name': 'SMD_Ori_Pikled', 'index': "2-1"},
    {'dataset_name': 'SMD_Ori_Pikled', 'index': "3-1"},
]

REAL_WORLD_CASE_NUM = len(real_world_configs)


def generate_real_world_dataset(case_idx=0, return_data=True):
    def get_config(case_idx):
        if not isinstance(case_idx, int):
            raise ValueError(f"case_idx must be an integer, got {type(case_idx)}. And it should be between 0 and {REAL_WORLD_CASE_NUM - 1}.")
        if case_idx < 0 or case_idx >= REAL_WORLD_CASE_NUM:
            raise ValueError(f"Invalid case index, case_idx={case_idx}. It should be between 0 and {REAL_WORLD_CASE_NUM - 1}.")
        return real_world_configs[case_idx]

    
    config = get_config(case_idx)
    # Here you would implement the logic to generate the real-world dataset based on the config
    # For now, we just return the config as a placeholder
    dataset_name = config['dataset_name']
    index_ = config.get('index', 1)
    data_pth_ = os.path.join(data_pth, dataset_name)
    dataset = get_loader_segment(index_, data_path=data_pth_, batch_size=100, win_size=100, step=100, dataset=dataset_name, ret_data=return_data)
    if config.get('index', None) is not None:
        dataset_name_new = f"SMD-{index_}"
    else:
        dataset_name_new = f"{dataset_name}"
    
    if return_data:
        train_x, test_x, test_y = dataset.load_data()
        return dataset_name_new, train_x, test_x, test_y
    else:
        return dataset_name_new, dataset
    