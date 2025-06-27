import sys
sys.path.append('..')
sys.path.append('../DeepOD')
sys.path.append('/home/zzj/Pycharm_project/DeepOD')

import os

os.chdir('/home/zzj/Pycharm_project/DeepOD')

import numpy as np
import matplotlib.pyplot as plt

# 设置参数
num_points = 1000  # 时间点的数量
my_th = 0.95
# my_th = 0.05
debug=0
typs=3
# 0, uniform_noise
# 1, gaussian_noise
# 2, model
prob=1

def plot_y(label):
    starts = np.where(np.diff(label) > 0)[0]
    ends = np.where(np.diff(label) < 0)[0]
    index1 = (0, len(label))
    test_data = np.arange(len(label))
    min_val = 0
    max_val = 1

    # min_val = np.min(test_data[index[0]:index[1]][:,data_col_index])
    # max_val = np.max(test_data[index[0]:index[1]][:,data_col_index])
    # print(starts, ends)

    # add subplot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    for start, end in zip(starts, ends):
        plt.fill_betweenx([min_val, max_val], start, end, color='red', alpha=0.2)

    plt.xlim(0, num_points)
        
    
# 定义生成连续段的函数
def generate_continuous_segments(num_segments, max_length, min_length=1):
    # signal = np.random.randint(0, 2, size=num_points)
    
    signal = np.zeros(num_points)
    for _ in range(num_segments):
        start = np.random.randint(0, len(signal) - max_length)
        end = start + max(np.random.randint(1, max_length + 1), min_length)
        signal[start:end] = 1
    return signal


# Demo1
def demo1(typs=0, prob=0.1,data_set=0):
    # generate borlin noise
    if data_set == 0:
        num_segments = 5
        max_length = 12
        min_length = 10
    
    elif data_set == 1:
        num_segments = 1
        max_length = 60
        min_length = 50

    elif data_set == 2:
        num_segments = 1
        max_length = 350
        min_length = 300
    y_test = generate_continuous_segments(num_segments=num_segments, max_length=max_length, min_length=min_length)
    if debug:
        plot_y(y_test)
    anomaly_scores = get_scores(y_test, typs, prob)
    thresh = np.quantile(anomaly_scores, my_th)
    pred_labels = (anomaly_scores > thresh).astype(int)
    if debug:
        plt.subplot(2, 1, 2)
        plt.plot(anomaly_scores)
        # plot y=thresh
        plt.plot([0, num_points], [thresh, thresh], 'r--')

        plt.xlim(0, num_points)
        plt.show()
    scores_list = combine_all_evaluation_scores_with_bias(pred_labels, y_test, anomaly_scores)
    return scores_list


eval_num = 20
if debug:
    eval_num = 1

scores_tests = ["f1_score_ori", "precision", "recall", "accuracy", "f1_score_pa", 
"point_auc",
"R_AUC_ROC",
"R_AUC_PR",
"VUS_ROC",
"VUS_PR",
"Affiliation precision",
"Affiliation recall",
"Aff_F1",
"NAff_F1",
"UAff_F1"]

out_pth = r'/home/zzj/Pycharm_project/DeepOD/metrics'

from collections import defaultdict
from tqdm import tqdm

