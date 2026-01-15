#!/usr/bin/env python3
from tigramite.toymodels import structural_causal_processes as toys
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

def lin_f(x):
    return x

def create_dataset_1_baseline():
    np.random.seed(123)
    links_coeffs = {
        0: [((0, -1), 0.8, lin_f)],
        1: [((1, -1), 0.77, lin_f)],
        2: [((0, -1), 0.65, lin_f),((1, -1),-0.79, lin_f)],
        3: [((3, -1), 0.65, lin_f)],
        4: [((1, -1), -0.3, lin_f),((2, -1),0.4, lin_f),((3, -1),0.55, lin_f)]
    }
    T = 1000
    data, _ = toys.structural_causal_process(links_coeffs, T=T, seed=123)
    df = pd.DataFrame(data, columns=['Y1', 'Y2', 'Y3', 'Y4', 'Y5'])
    dates = pd.date_range(start='2000-01-01', periods=T, freq='D')
    df.insert(0, 'date', dates)
    return df, links_coeffs

def create_dataset_2_more_noise():
    np.random.seed(124)
    links_coeffs = {
        0: [((0, -1), 0.8, lin_f)],
        1: [((1, -1), 0.77, lin_f)],
        2: [((0, -1), 0.65, lin_f),((1, -1),-0.79, lin_f)],
        3: [((3, -1), 0.65, lin_f)],
        4: [((1, -1), -0.3, lin_f),((2, -1),0.4, lin_f),((3, -1),0.55, lin_f)]
    }
    T = 1000
    data, _ = toys.structural_causal_process(links_coeffs, T=T, seed=124)
    np.random.seed(125)
    data[:, 0] += np.random.normal(0, 0.5, T)
    data[:, 1] += np.random.normal(0, 0.5, T)
    df = pd.DataFrame(data, columns=['Y1', 'Y2', 'Y3', 'Y4', 'Y5'])
    dates = pd.date_range(start='2000-01-01', periods=T, freq='D')
    df.insert(0, 'date', dates)
    return df, links_coeffs

def create_dataset_3_add_x0():
    np.random.seed(126)
    links_coeffs = {
        0: [((0, -1), 0.75, lin_f)],
        1: [((1, -1), 0.8, lin_f)],
        2: [((2, -1), 0.77, lin_f), ((0, -1), 0.4, lin_f)],
        3: [((1, -1), 0.65, lin_f),((2, -1),-0.79, lin_f)],
        4: [((4, -1), 0.65, lin_f)],
        5: [((2, -1), -0.3, lin_f),((3, -1),0.4, lin_f),((4, -1),0.55, lin_f)]
    }
    T = 1000
    data, _ = toys.structural_causal_process(links_coeffs, T=T, seed=126)
    df = pd.DataFrame(data, columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5'])
    dates = pd.date_range(start='2000-01-01', periods=T, freq='D')
    df.insert(0, 'date', dates)
    return df, links_coeffs

def create_dataset_4_add_x4a():
    np.random.seed(127)
    links_coeffs = {
        0: [((0, -1), 0.8, lin_f)],
        1: [((1, -1), 0.77, lin_f)],
        2: [((0, -1), 0.65, lin_f),((1, -1),-0.79, lin_f)],
        3: [((3, -1), 0.65, lin_f)],
        4: [((1, -1), -0.3, lin_f),((2, -1),0.4, lin_f),((3, -1),0.55, lin_f),((5, -1),0.35, lin_f)],
        5: [((5, -1), 0.7, lin_f)]
    }
    T = 1000
    data, _ = toys.structural_causal_process(links_coeffs, T=T, seed=127)
    df = pd.DataFrame(data, columns=['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y4a'])
    dates = pd.date_range(start='2000-01-01', periods=T, freq='D')
    df.insert(0, 'date', dates)
    return df, links_coeffs

def plot_datasets(datasets, dataset_names, output_folder='./datasets/'):
    os.makedirs(output_folder, exist_ok=True)
    for i, (df, name) in enumerate(zip(datasets, dataset_names)):
        data_cols = [col for col in df.columns if col != 'date']
        fig, axes = plt.subplots(len(data_cols), 1, figsize=(12, 2*len(data_cols)))
        fig.suptitle(f'{name}', fontsize=16)
        if len(data_cols) == 1:
            axes = [axes]
        for j, col in enumerate(data_cols):
            axes[j].plot(df[col].values[:200])
            axes[j].set_ylabel(col)
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'dataset{i+1}_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_datasets(datasets, dataset_names, links_coeffs_list, output_folder='./datasets/'):
    os.makedirs(output_folder, exist_ok=True)
    for i, (df, name, links_coeffs) in enumerate(zip(datasets, dataset_names, links_coeffs_list)):
        df.to_csv(os.path.join(output_folder, f'dataset{i+1}.csv'), index=False)
        with open(os.path.join(output_folder, f'dataset{i+1}_structure.txt'), 'w') as f:
            f.write(f"Dataset: {name}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Structural Causal Model:\n")
            f.write("-" * 25 + "\n")
            variables = [col for col in df.columns if col != 'date']
            for var_idx, var_name in enumerate(variables):
                if var_idx in links_coeffs:
                    f.write(f"{var_name}(t) = ")
                    terms = []
                    for (source_var, lag), coeff, func in links_coeffs[var_idx]:
                        source_name = variables[source_var] if source_var < len(variables) else f"Y{source_var}"
                        terms.append(f"{coeff} * {source_name}(t{lag})")
                    f.write(" + ".join(terms) + " + noise\n")
                else:
                    f.write(f"{var_name}(t) = noise only\n")
            f.write(f"\nDataset shape: {df.shape}\n")
            f.write(f"Variables: {variables}\n")
            f.write(f"Time series length: {len(df)}\n")

if __name__ == "__main__":
    df1, links1 = create_dataset_1_baseline()
    df2, links2 = create_dataset_2_more_noise()
    df3, links3 = create_dataset_3_add_x0()
    df4, links4 = create_dataset_4_add_x4a()

    datasets = [df1, df2, df3, df4]
    dataset_names = ["Dataset 1 Baseline", "Dataset 2 More Noise", "Dataset 3 Add X0->X1", "Dataset 4 Add X4a->X4"]
    links_coeffs_list = [links1, links2, links3, links4]

    save_datasets(datasets, dataset_names, links_coeffs_list)
    plot_datasets(datasets, dataset_names)
