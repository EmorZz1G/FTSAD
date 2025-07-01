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

import pandas as pd
def exec_config2table(configs, out_pth):
    """
    Convert configuration dictionary to a DataFrame and save it as a CSV file.
    
    Args:
        configs (dict): Dictionary containing configuration parameters.
        out_pth (str): Output path for the CSV file.
    """
    df = pd.DataFrame.from_dict(configs, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'config_name'}, inplace=True)
    df.columns = ['Case Name', 'TS Length', 'Segments', 'Max Seq Length', 'Max Seq Length']
    df.to_csv(out_pth, index=False)
    print(f"Configuration table saved to {out_pth}")


if __name__ == "__main__":
    out_pth = r'/home/zzj/projects/FTSAD/supp_files/config_table.csv'
    exec_config2table(configs, out_pth)
    print("Configuration table created successfully.")