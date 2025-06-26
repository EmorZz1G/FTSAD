import sys
import os

file_pth = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(os.path.dirname(file_pth))  # 获取上上级路径
dataset_dir = os.path.join(proj_dir, 'datasets')  # 数据集目录

sys.path.append(proj_dir)  # 将项目根目录添加到系统路径中
from src.data_utils.SimAD_data_loader2 import MSLSegLoader,SMAPSegLoader

import numpy as np

data_path = os.path.join(dataset_dir, 'SMAP')
train_X, test_X, test_y = SMAPSegLoader(data_path, 100, 100).load_data()

print("Test labels shape:", test_y.shape)
from src.metrics import basic_metricor

import unittest

class TestMetric(unittest.TestCase):
    def test_basic_metrics(self):
        metricor = basic_metricor()
        metricor.cal_unbiased_aff_prec_bias(test_y)
        score = np.random.rand(test_y.shape[0])  # 模拟预测分数
        q = 0.95
        pred = metricor.get_pred(score, quantile=q)
        res = metricor.metric_UN_Affiliation(test_y, score, pred=pred)
        print(res)


if __name__ == '__main__':
    unittest.main()