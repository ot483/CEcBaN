#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 11:06:04 2023

@author: ofir
"""
import sys
import numpy as np
import pandas as pd
import pickle 
from multiprocessing import Pool
import scipy
import pyEDM

BaseFolder = sys.argv[1]
cores = int(sys.argv[2])
max_mi_shift = int(sys.argv[3])
embedding_dim = int(sys.argv[4])
lag = int(sys.argv[5])
eccm_window_size = int(sys.argv[6])

def calcECCM(x_1, x_2, L, E):
    df = pd.DataFrame()
    df['x1'] = x_1
    df['x2'] = x_2
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < E + 1:
        print(f"Not enough data points: {len(df)} < {E + 1}")
        return 0, 0
    
    libstart = 1
    libend = int(len(df) * 0.75)
    predstart = libend + 1
    predend = len(df)
    
    if predstart >= len(df):
        predstart = libend
        predend = len(df)
    
    len_tr = libend - libstart + 1
    print("len_tr " + str(len_tr))
    
    if len_tr < 10:
        lib_lens = [max(E, len_tr)]
    else:
        lib_lens = list(range(max(10, E), min(len_tr, libend), 1))
    
    if not lib_lens:
        lib_lens = [max(E, 10)]
    
    try:
        ccm_result_x1 = pyEDM.CCM(
            dataFrame=df.reset_index(),
            E=E,
            Tp=0,
            columns='x1',
            target='x2',
            libSizes=lib_lens,
            sample=1,
            showPlot=False
        )
        
        ccm_result_x2 = pyEDM.CCM(
            dataFrame=df.reset_index(),
            E=E,
            Tp=0,
            columns='x2',
            target='x1',
            libSizes=lib_lens,
            sample=1,
            showPlot=False
        )
        
        sc1 = None
        sc2 = None
        
        if 'x1:x2' in ccm_result_x1:
            sc1 = ccm_result_x1['x1:x2']
        elif 'rho' in ccm_result_x1:
            sc1 = ccm_result_x1['rho']
        else:
            cols = [c for c in ccm_result_x1.keys() if c != 'LibSize']
            if cols:
                sc1 = ccm_result_x1[cols[0]]
        
        if 'x2:x1' in ccm_result_x2:
            sc2 = ccm_result_x2['x2:x1']
        elif 'rho' in ccm_result_x2:
            sc2 = ccm_result_x2['rho']
        else:
            cols = [c for c in ccm_result_x2.keys() if c != 'LibSize']
            if cols:
                sc2 = ccm_result_x2[cols[0]]
        
        if sc1 is None or sc2 is None:
            print("Could not extract scores from CCM results")
            return 0, 0
        
        df_Scores = pd.DataFrame()
        df_Scores["Library length"] = ccm_result_x1["LibSize"]
        df_Scores["x1"] = sc1
        df_Scores["x2"] = sc2
        
        df_Scores = df_Scores.set_index("Library length")
        df_Scores = df_Scores.fillna(0)
        
        print(df_Scores)
        
        Score_X1 = df_Scores["x1"].values[-5:].mean() if len(df_Scores["x1"].values) >= 5 else df_Scores["x1"].mean()
        Score_X2 = df_Scores["x2"].values[-5:].mean() if len(df_Scores["x2"].values) >= 5 else df_Scores["x2"].mean()
        
        return Score_X1, Score_X2
        
    except Exception as e:
        print(f"pyEDM CCM calculation error: {e}")
        return 0, 0

BaseFolder = sys.argv[1]
cores = int(sys.argv[2])

with open(BaseFolder + 'eccm_edges.pickle', 'rb') as handle:
    edges = pickle.load(handle)

with open(BaseFolder + 'eccm_dataset.pickle', 'rb') as handle:
    df_upsampled_proc = pickle.load(handle)

def manipulate(j):
    
    ll=2
    ee=5
    allECCM = []
     
    tmp_results = []
    for i in list(range(-1*max_mi_shift, max_mi_shift, 1)): 
      
            x1 = df_upsampled_proc[j[0]][-1*eccm_window_size:].values.tolist() #k[j[0]].values.tolist()  #j[0]
            x2 = df_upsampled_proc[j[1]][-1*eccm_window_size:].values.tolist() #k[j[1]].values.tolist() #j[1]
            
            #x1 -> x2_shifted 
            df_tmp = pd.DataFrame()
            df_tmp["x1"] = x1
            df_tmp["x2"] = x2
            df_tmp["x2_shifted"] = df_tmp["x2"].to_frame().shift(periods=i)
            if i < 0:
                df_tmp = df_tmp[:i]
            if i > 0:
                df_tmp = df_tmp[i:]
                
            s1_x2Shifted, s2_x2Shifted = calcECCM(x_1=df_tmp["x1"].copy(),
                                                x_2=df_tmp["x2_shifted"].copy(),
                                                L=ll,
                                                E=ee)        
            
            #x2 -> x1_shifted
            df_tmp = pd.DataFrame()
            df_tmp["x1"] = x1
            df_tmp["x2"] = x2        
            df_tmp["x1_shifted"] = df_tmp["x1"].to_frame().shift(periods=i)
            if i < 0:
                df_tmp = df_tmp[:i]
            if i > 0:
                df_tmp = df_tmp[i:]
                
            s1_x1Shifted, s2_x1Shifted = calcECCM(x_1=df_tmp["x2"].copy(),
                                                    x_2=df_tmp["x1_shifted"].copy(),
                                                    L=ll,
                                                    E=ee)                
            
            tmp_results.append([i, s1_x2Shifted, s1_x1Shifted])
            
    df_ = pd.DataFrame(tmp_results, columns=["l", "x1", "x2"]).set_index("l")
    
    df_["x1"] = df_["x1"].rolling(5, min_periods=1, center=True).mean()
    df_["x2"] = df_["x2"].rolling(5, min_periods=1, center=True).mean()
    
    
    
   
    max_arg_x1 = 0
    max_arg_x2 = 0
    
    negative_x1 = df_[df_.index <= 0]['x1']
    negative_x2 = df_[df_.index <= 0]['x2']
    
    if len(negative_x1) > 0:
        max_arg_x1 = negative_x1.idxmax()
    else:
        max_arg_x1 = df_['x1'].idxmax()
        
    if len(negative_x2) > 0:
        max_arg_x2 = negative_x2.idxmax()
    else:
        max_arg_x2 = df_['x2'].idxmax()
   
    allECCM.append([j[0], max_arg_x1, j[1], max_arg_x2, df_])
    print("*************************************")
    print(str(i))
    print("*************************************")    
    fig = df_.plot(title=str(j[0]) + "(x1) affects " + str(j[1]) + "(x2)")
    fig.set_xlabel('lag (timesteps)') 
    fig.set_ylabel('rho (p)') 
    fig.get_figure().savefig(BaseFolder + "eccm_" + str(j[0]) + "_" + str(j[1]) + ".png")
   
    return allECCM    

pool = Pool(cores)   

results_list_final = []
results_list_final = pool.map(manipulate, edges)

results_list_fixed = []
for i in results_list_final:
    results_list_fixed = results_list_fixed + i

pool.close()
pool.join()
print('end')
    
with open(BaseFolder+'AllECCM_results.pickle', 'wb') as handle:
      pickle.dump(results_list_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)