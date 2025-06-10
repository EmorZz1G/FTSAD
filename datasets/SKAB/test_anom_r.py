import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置数据集路径
pth1 = r'/workspaces/FTSAD/datasets/SKAB/other'
pth2 = r'/workspaces/FTSAD/datasets/SKAB/valve1'
pth3 = r'/workspaces/FTSAD/datasets/SKAB/valve2'

anoms = []
def test_pth(pth):
    global anoms
    for file in os.listdir(pth):
        print(file)
        if file.endswith('.csv'):
            file_path = os.path.join(pth, file)
            df = pd.read_csv(file_path, sep=';')
            print(df.shape)
            # 检查是否有 'anomaly' 列
            if 'anomaly' in df.columns:
                # 计算异常值的数量
                anom = df['anomaly'].values
                anom = sum(anom==1)/len(anom)*100
                print(f"File: {file}, Anomaly Ratio: {anom}")
                anoms.append(anom)
                # 异常比率
                # anomaly_ratio = anomaly_count / len(df) * 100
                # print(f"File: {file}, Anomaly Count: {anomaly_count}, Anomaly Ratio: {anomaly_ratio:.2%}")
        
test_pth(pth1)
test_pth(pth2)
test_pth(pth3)

anom = np.mean(anoms)
print(f"Average Anomaly Ratio: {anom:.2f}%")

