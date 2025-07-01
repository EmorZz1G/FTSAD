import os
from os.path import dirname as upd
file_pth = os.path.dirname(os.path.abspath(__file__))
file_pth = upd(upd(file_pth))
proj_pth = upd(file_pth)
data_pth = os.path.join(proj_pth, 'datasets')
import sys
sys.path.append(proj_pth)

import numpy as np
import random

real_world_configs = [
    {'dataset_name': 'MSL'},
    {'dataset_name': 'NIPS_TS_Creditcard'},
    {'dataset_name': 'SWAT'},
    {'dataset_name': 'SMD_Ori_Pikled', 'index': "1-1"},
    {'dataset_name': 'SMD_Ori_Pikled', 'index': "2-1"},
    {'dataset_name': 'SMD_Ori_Pikled', 'index': "3-1"},
]

REAL_WORLD_CASE_NUM = len(real_world_configs)

from src.data_utils.SimAD_data_loader2 import get_loader_segment
from itertools import groupby
from operator import itemgetter

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
    


def convert_vector_to_events(vector = [0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).
    
    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix : ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    
    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y+1) for (x,y) in events]
        
    return (events)

def test_dataset(dataset_name, train_x, test_x, test_y):
    anom_ratio = np.sum(test_y) / len(test_y)
    anom_ranges = convert_vector_to_events(test_y)

    anoms_len = []
    for st, ed in anom_ranges:
        anoms_len.append(ed - st)

    mx_anom_len = max(anoms_len) if anoms_len else 0
    min_anom_len = min(anoms_len) if anoms_len else 0
    avg_anom_len = np.mean(anoms_len) if anoms_len else 0
    test_len = len(test_y)
    segments = len(anom_ranges)
    res = {
        'Dataset': dataset_name,
        'TS Length': test_len,
        "Segments": segments,
        "Max Seg Length": mx_anom_len,
        "Min Seg Length": min_anom_len,
        # 'Anomaly Ratio': anom_ratio*100,
    }
    print(f"Dataset: {dataset_name}, TS Length: {test_len}, Segments: {segments}, Max Seg Length: {mx_anom_len}, Min Seg Length: {min_anom_len}, Anomaly Ratio: {anom_ratio*100:.2f}%")
    return res

import pandas as pd
def test_dataset_attr():
    dat = []
    for i in range(REAL_WORLD_CASE_NUM):
        dataset_name, train_x, test_x, test_y = generate_real_world_dataset(i)
        assert isinstance(dataset_name, str), f"Dataset name should be a string, got {type(dataset_name)}"
        assert isinstance(train_x, np.ndarray), f"Train data should be a numpy array, got {type(train_x)}"
        assert isinstance(test_x, np.ndarray), f"Test data should be a numpy array, got {type(test_x)}"
        assert isinstance(test_y, np.ndarray), f"Test labels should be a numpy array, got {type(test_y)}"
        print(f"Test passed for dataset: {dataset_name}")
        res = test_dataset(dataset_name, train_x, test_x, test_y)
        dat.append(res)
    df = pd.DataFrame(dat)
    df.to_csv(os.path.join(r'/home/zzj/projects/FTSAD/supp_files', 'real_config_table.csv'), index=False)


if __name__ == "__main__":
    test_dataset_attr()
    print("All tests passed!")