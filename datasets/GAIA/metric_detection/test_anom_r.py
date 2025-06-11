pth = r'/workspaces/FTSAD/datasets/GAIA/'
import os

import pandas as pd


cnt = 0
ts_len = []
ts_anom = []
for root , subdir, file in os.walk(pth):
    # print(root,subdir,file)
    if file and file[0] == '.DS_Store':
        continue
    if len(file)>1:
        cnt += len(file)
        for f in file:
            print(root,f)
            dat = pd.read_csv(os.path.join(root,f))
            # print(dat.head(),dat.shape)
            ts_len.append(dat.shape[0])
            lb = dat['label']
            anom = sum(lb==1)/len(dat)
            
            ts_anom.append(anom)
            
import numpy as np

ts_len = np.mean(ts_len)
ts_anom = np.mean(ts_anom)
print(cnt,ts_len,ts_anom)        