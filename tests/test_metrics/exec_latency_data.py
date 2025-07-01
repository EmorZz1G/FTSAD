pth_fix = r'/home/zzj/projects/FTSAD/logs/latency_log_fix.csv'

import pandas as pd
df = pd.read_csv(pth_fix, sep=',', header=0)

tmp = df.groupby(['case_name','model_name','case_seed','metric_name'])

tmp1 = tmp['val'].agg(['mean', 'std']).add_suffix('_val')
tmp2 = tmp['latency'].agg(['mean', 'std']).add_suffix('_latency')

# 将聚合结果合并到一个新的DataFrame中
result = tmp1.join(tmp2)

# 重置索引，将分组键从索引变为列
result = result.reset_index()

out_pth = r'/home/zzj/projects/FTSAD/supp_files/latency_case_data_processed.csv'
result.to_csv(out_pth, index=False)



# format for visualization in Table

def get_vis(mean_vals, std_vals):
    return [f"{mean*100:.2f} ± {std*100:.2f}" for mean, std in zip(mean_vals, std_vals)]

def get_vis_latency(mean_vals, std_vals):
    return [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(mean_vals, std_vals)]

df2 = result.copy()
df2.drop(columns=['case_seed'], inplace=True)
print(df2.head())
df2['Score'] = get_vis(df2['mean_val'].values, df2['std_val'].values)

df2.drop(columns=['mean_val', 'std_val'], inplace=True)
df2['Latency'] = get_vis_latency(df2['mean_latency'].values, df2['std_latency'].values)

df2.drop(columns=['mean_latency', 'std_latency'], inplace=True)
df2.rename(columns={'case_name': 'Dataset', 'model_name': 'Model', 'metric_name': 'Metric'}, inplace=True)

out_pth = r'/home/zzj/projects/FTSAD/supp_files/latency_case_data_vis.csv'
df2.to_csv(out_pth, index=False)
print(df2.head())
# Save the processed DataFrame to a new CSV file