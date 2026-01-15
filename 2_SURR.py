#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ofir
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyEDM
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests
import argparse


def make_stationary(column):
    adf_result = adfuller(column)
    p_value = adf_result[1]
    if p_value >= 0.05: 
        diff_column = column.diff() 
        return diff_column
    else:
        return column


def amplifyData(df, subSetLength=600, jumpN=30):
    allDfs = []
    for i in list(range(1, len(df)-subSetLength, jumpN)):
        tmp = df.iloc[i:i+subSetLength]
        allDfs.append(tmp)
    return allDfs

def build_colsDict(df):
    dd = {}
    counter = 0
    for i in df.columns:
        counter=counter+1
        dd[i] = "col_"+str(counter)
        dd["col_"+str(counter)] = i
    return dd





parser = argparse.ArgumentParser(description="Process data for CCM and ECCM analysis.")
parser.add_argument('--file_path', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output_folder', type=str, required=True, help='Path to tunique output folder each run')    
parser.add_argument('--subSetLength', type=int, default=60, help='Subset length for analysis')
parser.add_argument('--jumpN', type=int, default=30, help='Jump N value for processing')
parser.add_argument('--z_score_threshold', type=float, default=3.0, help='Z-score threshold for outlier detection')
parser.add_argument('--resample_freq', type=str, default='1M', help='Frequency for data resampling')
parser.add_argument('--embedding_dim', type=int, default=2, help='Embedding dimension for CCM')
parser.add_argument('--lag', type=int, default=1, help='Lag for CCM')
parser.add_argument('--number_of_cores', type=int, default=1, help='Number of cores for multithreading')
parser.add_argument('--ccm_training_proportion', type=float, default=0.75, help='ccm_training_proportion in CCM calculation')
parser.add_argument('--max_mi_shift', type=int, default=20, help='max_mutual_information_shift')
parser.add_argument('--num_surrogates_x1', type=int, default=10, help='Number of surrogates for x1')
parser.add_argument('--num_surrogates_x2', type=int, default=10, help='Number of surrogates for x2')
parser.add_argument('--sig_quant', type=float, default=0.9, help='Significance quantile')
parser.add_argument('--preserve_manual_edits', type=str, default='False', help='Preserve manual edits flag')


args = parser.parse_args()

file_path = str(args.file_path)
subSetLength = int(args.subSetLength)
jumpN = int(args.jumpN)
z_score_threshold = float(args.z_score_threshold)
resample_freq = str(args.resample_freq)
embedding_dim = int(args.embedding_dim)
lag = int(args.lag)
number_of_cores = int(args.number_of_cores)
ccm_training_proportion = float(args.ccm_training_proportion)
max_mi_shift = int(args.max_mi_shift)
num_surrogates_x1 = int(args.num_surrogates_x1)
num_surrogates_x2 = int(args.num_surrogates_x2)
sig_quant = float(args.sig_quant)
preserve_manual_edits = args.preserve_manual_edits == 'True'
output_folder = args.output_folder

if not output_folder.endswith(os.sep):
    output_folder = output_folder + os.sep

print(f"Output folder set to: {output_folder}")

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

BaseFolder = "./"

concated_ = load_data(file_path)
print("processing data....")

# Check if manual edits should be preserved
manual_edits_flag = os.path.join(output_folder, 'MANUAL_EDITS_MADE.flag')
if preserve_manual_edits and os.path.exists(manual_edits_flag):
    print("Manual edits detected - preserving user changes")
    # Work with a copy to avoid overwriting manual edits
    curated_backup = os.path.join(output_folder, 'CCM_ECCM_curated_original.csv')
    curated_main = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
    if not os.path.exists(curated_backup):
        # Create backup of manually edited version
        import shutil
        shutil.copy(curated_main, curated_backup)
        print(f"Created backup of manual edits: {curated_backup}")
        
try:
    concated_['Date'] = pd.to_datetime(concated_['Date'])
    concated_ = concated_.set_index('Date')
except:
    concated_['date'] = pd.to_datetime(concated_['date'])
    concated_ = concated_.set_index('date')
    
cols = list(concated_.columns)

df_interpolatedNotNormalized = concated_.copy()

#Normalize 0-1
df_upsampled_normalized = pd.DataFrame(index = concated_.index)
#df_upsampled_normalized = df_concated_smoothed.copy()
AllScalersDict = {}
for i in concated_.columns:
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(concated_[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_concated_fixed_outlayers = df_upsampled_normalized.copy()
df_concated_fixed_outlayers = df_concated_fixed_outlayers.resample(resample_freq).interpolate(method='linear') #its already 7 days, this interpolation is for the case there are missing values
df_concated_fixed_outlayers[df_concated_fixed_outlayers < 0] = 0

for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > z_score_threshold)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate(method='linear')
    
df_concated_fixed_outlayers[df_concated_fixed_outlayers < 0] = 0

df_concated_fixed_outlayers = df_concated_fixed_outlayers.dropna()



with open(os.path.join(output_folder, 'All_ccm2_results.pickle'), 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


df_CausalFeatures2 = pd.read_csv(os.path.join(output_folder, 'CCM_ECCM_curated.csv'))

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= max_mi_shift]

df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()

df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates(['x1', 'x2'])

Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))

os.environ['MKL_NUM_THREADS'] = '1'

x1x2s = df_CausalFeatures2[['x1', 'x2']].values.tolist()
x1x2s = [(i[0], i[1]) for i in x1x2s]

pairs = x1x2s
pairs = [(i[0], i[1]) for i in pairs]
pairs = list(set(pairs))

surr_file_path = os.path.join(output_folder, 'surr_results.pickle')

if os.path.exists(surr_file_path):
    os.remove(surr_file_path)
    print(f"File {surr_file_path} has been deleted.")
else:
    print(f"File {surr_file_path} does not exist.")
    
    
for p in pairs:
    
    try:
        with open(os.path.join(output_folder, 'surr_results.pickle'), 'rb') as handle:
            Dict_sur = pickle.load(handle)
    except:
        Dict_sur = {}   
        with open(os.path.join(output_folder, 'surr_results.pickle'), 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)      
            
    allKeys = Dict_sur.keys()
    if not p in allKeys:
        
        x = 0
        slc = 1
        while x == 0:
            try:
                x = x + 1
                df_x1 = pd.DataFrame(data = df_concated_fixed_outlayers[[p[0]]].values.tolist(), columns = [p[0]])
                df_x1_sliced = df_x1[[p[0]]][1:len(df_x1)-slc]
                df_sur_x1 = pyEDM.SurrogateData(dataFrame=df_x1_sliced, column=p[0], method='ebisuzaki', numSurrogates=num_surrogates_x1)
            except:
                x = 0
                slc += 1

        x = 0
        slc = 1
        while x == 0:
            try:
                x = x + 1
                df_x2 = pd.DataFrame(data = df_concated_fixed_outlayers[[p[1]]].values.tolist(), columns = [p[1]])
                df_x2_sliced = df_x2[[p[1]]][1:len(df_x2)-slc]
                df_sur_x2 = pyEDM.SurrogateData(dataFrame=df_x2_sliced, column=p[1], method='ebisuzaki', numSurrogates=num_surrogates_x2)
            except:
                x = 0
                slc += 1

        
        
        Dict_sur[(p[0], p[1])] = []   
        sur_cols_x1 = list(df_sur_x1.columns)[1:]
        sur_cols_x2 = list(df_sur_x2.columns)[1:]
        
        df_suf = pd.DataFrame(index=df_concated_fixed_outlayers[1:].index)
        df_suf = pd.concat([df_sur_x1[sur_cols_x1], df_sur_x2[sur_cols_x2]], axis=1)
        
        amplified_dfs = amplifyData(df_suf, subSetLength=subSetLength, jumpN=jumpN)
        DictCols = build_colsDict(df_suf)
    
        for i, val in enumerate(amplified_dfs):
            val.columns = [DictCols[i] for i in val.columns]
            
            for col in val.columns:
                val[col] = make_stationary(val[col])
            
            amplified_dfs[i] = val
        
        with open(os.path.join(output_folder, 'surr_amplified_dfs.pickle'), 'wb') as handle:
            pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        with open(os.path.join(output_folder, 'surr_DictCols.pickle'), 'wb') as handle:
            pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        with open(os.path.join(output_folder, 'surr_x1_x2_columns.pickle'), 'wb') as handle:
            pickle.dump([sur_cols_x1, sur_cols_x2], handle, protocol=pickle.HIGHEST_PROTOCOL)      

        os.system('python '+BaseFolder+'/scripts/ccm_multiproc_1.py '+ output_folder + ' surr_ ' + str(number_of_cores) +\
                                                                                              " " + str(max_mi_shift) +\
                                                                                              " " + str(embedding_dim) +\
                                                                                              " " + str(lag) +\
                                                                                              " " + str(0))    
        with open(os.path.join(output_folder, 'All_surr_results.pickle'), 'rb') as handle:
           results_list_fixed = pickle.load(handle)   
                
        res = results_list_fixed
        tmp = []
        AllSurr=[]
        for j in res:
            try:
                s = j[1] 
                tmp.append([p[0], p[1], s])
            except:
                print('e')
               
        for j in tmp:
            try:
                s = j[2]['x1_mean'][-10:].mean()
                AllSurr.append([p[0], p[1], s])
            except:
                AllSurr.append([p[0], p[1], 0])
               
        Dict_sur[(p[0], p[1])].append(AllSurr)
    
        with open(os.path.join(output_folder, 'surr_results.pickle'), 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(os.path.join(output_folder, 'surr_results.pickle'), 'rb') as handle:
    Dict_sur = pickle.load(handle)

df_CausalFeatures2 = pd.read_csv(os.path.join(output_folder, 'CCM_ECCM_curated.csv'))

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= max_mi_shift]

AllSurr = []
for i in Dict_sur.keys():
    if i in pairs:
        tmp = []
        for j in Dict_sur[i][0]:
            try:
                s = j[2] 
                tmp.append([i[0], i[1], s])
            except:
                print('e')
        for j in tmp:
            s = j[2] 
            AllSurr.append([i[0], i[1], s])
        

df_AllSurr = pd.DataFrame(data=AllSurr, columns=['x1', 'x2', 'Score'])
df_AllSurr['x1x2'] = df_AllSurr['x1']+"_"+df_AllSurr['x2']

df_truth = df_CausalFeatures2[['x1', 'x2', 'Score']]
df_truth = df_truth.reset_index()
df_truth['x1x2'] = df_truth['x1']+"_"+df_truth['x2']
df_truth = df_truth.groupby('x1x2', group_keys=False).apply(lambda x: x.loc[x.Score.idxmax()])
df_truth = df_truth.set_index("index").reset_index()

Dict_quantiles = {}
All_quantiles90 = []
All_quantiles95 = []
All_quantiles975 = []

for i in df_AllSurr["x1x2"].unique():
    arr = df_AllSurr[df_AllSurr["x1x2"] == i]["Score"].values    
    q90 = np.quantile(arr, sig_quant)    
    Dict_quantiles[i] = q90
    All_quantiles90.append([i, q90])
  
df_quantiles90 = pd.DataFrame(data=All_quantiles90, columns=["x1x2", "Score"])
df_quantiles90 = df_quantiles90.reset_index()


# ============================================================================
# FIX 1.2: MULTIPLE TESTING CORRECTION (FDR - Benjamini-Hochberg)
# ============================================================================
print("\n" + "="*70)
print("CALCULATING EMPIRICAL P-VALUES AND FDR CORRECTION")
print("="*70)

# Calculate empirical p-values for each variable pair
pvalue_results = []

for pair in df_truth["x1x2"].unique():
    # Get observed score for this pair
    observed_score = df_truth[df_truth["x1x2"] == pair]["Score"].values

    if len(observed_score) == 0:
        print(f"  Warning: No observed score for pair {pair}")
        continue

    observed_score = observed_score[0]

    # Get all surrogate scores for this pair
    surrogate_scores = df_AllSurr[df_AllSurr["x1x2"] == pair]["Score"].values

    if len(surrogate_scores) == 0:
        print(f"  Warning: No surrogate scores for pair {pair}")
        continue

    # Calculate empirical p-value
    # p = (number of surrogates >= observed + 1) / (total surrogates + 1)
    # This is the standard formula that avoids p=0
    r = np.sum(surrogate_scores >= observed_score)
    n = len(surrogate_scores)
    p_value = (r + 1) / (n + 1)

    # Get x1 and x2 separately
    x1, x2 = pair.split('_', 1)

    pvalue_results.append({
        'x1': x1,
        'x2': x2,
        'pair': pair,
        'observed_score': observed_score,
        'n_surrogates': n,
        'n_exceeding': r,
        'p_value_raw': p_value
    })

# Create DataFrame
df_pvalues = pd.DataFrame(pvalue_results)

if len(df_pvalues) > 0:
    print(f"\nCalculated p-values for {len(df_pvalues)} variable pairs")
    print(f"Number of surrogates per pair: {df_pvalues['n_surrogates'].iloc[0]}")

    # Apply FDR correction (Benjamini-Hochberg procedure)
    # multipletests returns: reject, pvals_corrected, alphacSidak, alphacBonf
    alpha = 0.05
    reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
        df_pvalues['p_value_raw'].values,
        alpha=alpha,
        method='fdr_bh'
    )

    # Add corrected values to DataFrame
    df_pvalues['p_value_fdr'] = pvals_corrected
    df_pvalues['significant_raw'] = df_pvalues['p_value_raw'] < alpha
    df_pvalues['significant_fdr'] = reject

    # Calculate statistics
    n_sig_raw = df_pvalues['significant_raw'].sum()
    n_sig_fdr = df_pvalues['significant_fdr'].sum()
    n_rejected = n_sig_raw - n_sig_fdr

    # Sort by FDR-corrected p-value
    df_pvalues = df_pvalues.sort_values('p_value_fdr')

    # Save to CSV
    try:
        csv_path = output_folder + 'surrogate_pvalues_corrected.csv'
        df_pvalues.to_csv(csv_path, index=False)
        print(f"\n✓ Saved p-values to: {csv_path}")
    except Exception as e:
        print(f"\n!  Warning: Could not save p-values CSV: {e}")

    # Report results
    print("\n" + "="*70)
    print("FDR CORRECTION RESULTS")
    print("="*70)
    print(f"Total variable pairs tested: {len(df_pvalues)}")
    print(f"Significant pairs (raw p < {alpha}): {n_sig_raw}")
    print(f"Significant pairs (FDR-corrected): {n_sig_fdr}")
    print(f"False discoveries prevented: {n_rejected}")
    print(f"FDR correction rate: {n_rejected/n_sig_raw*100:.1f}%" if n_sig_raw > 0 else "N/A")

    # Show top 5 most significant pairs
    if len(df_pvalues) > 0:
        print("\nTop 5 most significant causal pairs (FDR-corrected):")
        print("-" * 70)
        top5 = df_pvalues.head(5)
        for idx, row in top5.iterrows():
            sig_marker = "✓" if row['significant_fdr'] else "✗"
            print(f"  {sig_marker} {row['x1']:20s} → {row['x2']:20s} | "
                  f"p_raw={row['p_value_raw']:.4f}, p_fdr={row['p_value_fdr']:.4f}")

    print("="*70)
else:
    print("\n!  Warning: No p-values calculated")

# ============================================================================
# END FDR CORRECTION
# ============================================================================


def save_surrogate_plot(output_folder, df_AllSurr, df_truth, df_quantiles90):
    """
    Robust function to create and save surrogate plot
    """
    try:
        if df_AllSurr.empty:
            print("WARNING: df_AllSurr is empty - cannot create plot")
            return False
            
        if df_truth.empty:
            print("WARNING: df_truth is empty - cannot create plot")
            return False
            
        if df_quantiles90.empty:
            print("WARNING: df_quantiles90 is empty - cannot create plot")
            return False
        
        
        fig = plt.figure(figsize=(25, 15))
        plt.rcParams.update({'font.size': 24})


        g = df_AllSurr.plot(kind="scatter", x="x1x2", y="Score", color='gray', alpha=0.5, s=100, marker="o", figsize=(25, 15))
        ax = df_truth.plot(kind="scatter", x="x1x2", y="Score", color='red', s=20, marker="o", ax=g)
        ax2 = df_quantiles90.plot(kind="scatter", x="x1x2", y="Score", color='black', s=100, marker="_", ax=ax)

        ax2.set_xticklabels(df_quantiles90["x1x2"], rotation=90)
        
        plot_path = os.path.join(output_folder, 'Surr_plot.png')
        
        plt.savefig(plot_path, 
                   bbox_inches='tight', 
                   transparent=False,
                   dpi=150,
                   format='png',
                   facecolor='white',
                   edgecolor='none')
        
        plt.close(fig)  
        
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            if file_size > 1000:  
                print(f"Surrogate plot saved successfully: {plot_path} ({file_size} bytes)")
                return True
            else:
                print(f"Plot file created but too small: {file_size} bytes")
                return False
        else:
            print("Plot file was not created")
            return False
            
    except Exception as e:
        print(f"Error creating surrogate plot: {e}")
        import traceback
        traceback.print_exc()
        return False

success = save_surrogate_plot(output_folder, df_AllSurr, df_truth, df_quantiles90)

if not success:
    print("Failed to create surrogate plot!")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Surrogate plot generation failed\nCheck data and try again', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fallback_path = os.path.join(output_folder, 'Surr_plot.png')
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Created fallback plot at: {fallback_path}")
    except Exception as e:
        print(f"Even fallback plot creation failed: {e}")

df_CausalFeatures2 = pd.read_csv(os.path.join(output_folder, 'CCM_ECCM_curated.csv'))

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= max_mi_shift]

df_CausalFeatures2['x1x2'] = df_CausalFeatures2['x1']+"_"+df_CausalFeatures2['x2']
try:
    del df_quantiles90["index"]
except:
    print()

# =============================================================================
df_quantiles90.columns = ["x1x2", "Score_quantile"]
df_quantiles90["Score_quantile"] = df_quantiles90["Score_quantile"].round(2)

df_CausalFeatures2 = pd.merge(df_CausalFeatures2, df_quantiles90, on="x1x2")

# Apply BOTH filters: quantile threshold (raw p-value) AND FDR significance
n_before = len(df_CausalFeatures2)

# Calculate the alpha threshold from sig_quant
# If sig_quant = 0.95, we want p_value <= 0.05 (top 5%)
# If sig_quant = 0.90, we want p_value <= 0.10 (top 10%)
alpha_quantile = 1.0 - sig_quant

print(f"\n--- SURROGATE FILTERING ---")
print(f"sig_quant = {sig_quant} → alpha = {alpha_quantile}")

if len(df_pvalues) > 0:
    # Merge p-values (both raw and FDR-corrected)
    df_pvalues_for_merge = df_pvalues[['pair', 'p_value_raw', 'p_value_fdr', 'significant_fdr']].copy()
    df_pvalues_for_merge.columns = ['x1x2', 'p_value_raw', 'p_value_fdr', 'significant_fdr']
    df_CausalFeatures2 = pd.merge(df_CausalFeatures2, df_pvalues_for_merge, on="x1x2", how="left")

    # Filter 1: Raw p-value must be <= alpha (score in top percentile as specified by sig_quant)
    quantile_mask = df_CausalFeatures2["p_value_raw"] <= alpha_quantile
    n_pass_quantile = quantile_mask.sum()
    print(f"Quantile filter (p_value_raw <= {alpha_quantile}): {n_pass_quantile}/{n_before} pass")

    # Filter 2: FDR significance
    fdr_mask = df_CausalFeatures2["significant_fdr"] == True
    n_pass_fdr = fdr_mask.sum()
    print(f"FDR filter (p_value_fdr < 0.05): {n_pass_fdr}/{n_before} pass")

    # Apply BOTH filters
    combined_mask = quantile_mask & fdr_mask
    df_CausalFeatures2 = df_CausalFeatures2[combined_mask]
    n_after = len(df_CausalFeatures2)

    print(f"Combined (BOTH must pass): {n_before} → {n_after} interactions ({n_before - n_after} removed)")
else:
    # Fallback to score-based quantile filtering if no p-values available
    print("\n!  No p-values available, using score threshold filtering only")
    quantile_mask = df_CausalFeatures2["Score"] >= df_CausalFeatures2["Score_quantile"]
    df_CausalFeatures2 = df_CausalFeatures2[quantile_mask]
    n_after = len(df_CausalFeatures2)
    print(f"Quantile only: {n_before} → {n_after} interactions ({n_before - n_after} removed)")

try:
    del df_CausalFeatures2['Unnamed: 0']
except:
    print()

df_CausalFeatures2.to_csv(os.path.join(output_folder, "Surr_filtered.csv"))
