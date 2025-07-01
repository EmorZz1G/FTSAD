pth = r'/home/zzj/projects/FTSAD/logs/latency_log.csv'
pth_fix = r'/home/zzj/projects/FTSAD/logs/latency_log_fix.csv'

import pandas as pd
df = pd.read_csv(pth, sep=',', header=0)
print(df.head())
# 找到metric_name列为F1-PA所有名字为的行
df_f1_pa = df[df['metric_name'] == 'F1-PA']
print(df_f1_pa.head())

def exc(x):
    print(x)
    # (0.9381863821775965, 0.9803, 0.8835697399527187, 1.0)
    if isinstance(x, str):
        x = eval(x)[0]
    print(x)
    return x


# 使用.map方法应用exc函数，并将结果赋值回原始DataFrame
df.loc[df['metric_name'] == 'F1-PA', 'val'] = df.loc[df['metric_name'] == 'F1-PA', 'val'].map(exc)

print(df.head())
df.to_csv(pth_fix, index=False)
