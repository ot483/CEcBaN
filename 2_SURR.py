#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ofir
"""
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import pyEDM
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
from statsmodels.tsa.stattools import adfuller


def make_stationary(column):
    adf_result = adfuller(column)
    p_value = adf_result[1]
    if p_value >= 0.05:  # If p-value is greater than or equal to 0.05, column is non-stationary
        diff_column = column.diff()  # Difference the column
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



import argparse



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
output_folder = args.output_folder


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

BaseFolder = "./"

concated_ = load_data(file_path)
print("processing data....")

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



#refine CCM network using eccm
#is_valid legend: 0-not valid; 1-both reponse to common strong force, at time 0 - synchrony; 2- x1 causes x2;
#extract time to effect

with open(output_folder + 'All_ccm2_results.pickle', 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


df_CausalFeatures2 = pd.read_csv(output_folder+'CCM_ECCM_curated.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= max_mi_shift]

##All_causal_CCM_dfs
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

# Check if the file exists
if os.path.exists(surr_file_path):
    os.remove(surr_file_path)
    print(f"File {surr_file_path} has been deleted.")
else:
    print(f"File {surr_file_path} does not exist.")
    
    
for p in pairs:
    
    try:
        with open(output_folder+'surr_results.pickle', 'rb') as handle:
            Dict_sur = pickle.load(handle)
    except:
        Dict_sur = {}   
        with open(output_folder+'surr_results.pickle', 'wb') as handle:
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
        #measure ccm and save score in a dict
        sur_cols_x1 = list(df_sur_x1.columns)[1:]
        sur_cols_x2 = list(df_sur_x2.columns)[1:]
        
        df_suf = pd.DataFrame(index=df_concated_fixed_outlayers[1:].index)
        df_suf = pd.concat([df_sur_x1[sur_cols_x1], df_sur_x2[sur_cols_x2]], axis=1)
        
        amplified_dfs = amplifyData(df_suf, subSetLength=subSetLength, jumpN=jumpN)#, FromYear=2001, ToYear=2020)
        DictCols = build_colsDict(df_suf)
    
        for i, val in enumerate(amplified_dfs):
            val.columns = [DictCols[i] for i in val.columns]
            
            for col in val.columns:
                val[col] = make_stationary(val[col])
            
            amplified_dfs[i] = val
        
        #save amplified df as pickle to be read by the external process
        with open(output_folder+'surr_amplified_dfs.pickle', 'wb') as handle:
            pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        with open(output_folder+'surr_DictCols.pickle', 'wb') as handle:
            pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        with open(output_folder+'surr_x1_x2_columns.pickle', 'wb') as handle:
            pickle.dump([sur_cols_x1, sur_cols_x2], handle, protocol=pickle.HIGHEST_PROTOCOL)      

        ##multiprocessing
        os.system('python '+BaseFolder+'/scripts/ccm_multiproc_1.py '+ output_folder + ' surr_ ' + str(number_of_cores) +\
                                                                                              " " + str(max_mi_shift) +\
                                                                                              " " + str(embedding_dim) +\
                                                                                              " " + str(lag) +\
                                                                                              " " + str(0))    
        with open(output_folder + 'All_surr_results.pickle', 'rb') as handle:
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
    
        with open(output_folder+'surr_results.pickle', 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
        
      
        
      
        
      
        
with open(output_folder+'surr_results.pickle', 'rb') as handle:
    Dict_sur = pickle.load(handle)

df_CausalFeatures2 = pd.read_csv(output_folder+'CCM_ECCM_curated.csv')


#Filter df_CausalFeatures2 by eccm
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

#Calculate quantile and check if it above 
for i in df_AllSurr["x1x2"].unique():
    arr = df_AllSurr[df_AllSurr["x1x2"] == i]["Score"].values    
    q90 = np.quantile(arr, sig_quant)    
    Dict_quantiles[i] = q90
    All_quantiles90.append([i, q90])
  
df_quantiles90 = pd.DataFrame(data=All_quantiles90, columns=["x1x2", "Score"])
df_quantiles90 = df_quantiles90.reset_index()




plt.figure(figsize=(25, 15))
plt.rcParams.update({'font.size': 24})

# Plot the data
g = df_AllSurr.plot(kind="scatter", x="x1x2", y="Score", color='gray', alpha=0.5, s=100, marker="o", figsize=(25, 15))
ax = df_truth.plot(kind="scatter", x="x1x2", y="Score", color='red', s=20, marker="o", ax=g)
ax2 = df_quantiles90.plot(kind="scatter", x="x1x2", y="Score", color='black', s=100, marker="_", ax=ax)

ax2.set_xticklabels(df_quantiles90["x1x2"], rotation=90)

plt.savefig(output_folder+'Surr_plot.png', bbox_inches='tight', transparent=False)
plt.close()

#Filter df_CausalFeatures2 by eccm
df_CausalFeatures2 = pd.read_csv(output_folder+'CCM_ECCM_curated.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= max_mi_shift]

#Filter df_CausalFeatures2 by quantile
df_CausalFeatures2['x1x2'] = df_CausalFeatures2['x1']+"_"+df_CausalFeatures2['x2']
try:
    del df_quantiles90["index"]
except:
    print()


# =============================================================================
df_quantiles90.columns = ["x1x2", "Score_quantile"]
df_quantiles90["Score_quantile"] = df_quantiles90["Score_quantile"].round(2)

df_CausalFeatures2 = pd.merge(df_CausalFeatures2, df_quantiles90, on="x1x2")
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] >= df_CausalFeatures2["Score_quantile"]]

try:
    del df_CausalFeatures2['Unnamed: 0']
except:
    print()




df_CausalFeatures2.to_csv(output_folder+"Surr_filtered.csv")




