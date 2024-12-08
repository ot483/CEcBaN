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
import skccm as ccm
from skccm.utilities import train_test_split





BaseFolder = sys.argv[1]
cores = int(sys.argv[2])
max_mi_shift = int(sys.argv[3])
embedding_dim = int(sys.argv[4])
lag = int(sys.argv[5])
eccm_window_size = int(sys.argv[6])




def calcECCM(x_1, x_2, L, E):
    e1 = ccm.Embed(x_1)
    e2 = ccm.Embed(x_2)

    X1 = e1.embed_vectors_1d(L,E)
    X2 = e2.embed_vectors_1d(L,E)
    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)
    
    CCM = ccm.CCM() #initiate the class
    
    #library lengths to test
    len_tr = len(x1tr)
    print("len_tr "+str(len_tr))
    #lib_lens = np.arange(10, len_tr, len_tr/2, dtype='int')
    lib_lens = list(range(10, len_tr, 1))
    
    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
    
    sc1,sc2 = CCM.score()
    
    df_Scores = pd.DataFrame()
    df_Scores["Library length"] = lib_lens
    df_Scores["x1"] = sc1
    df_Scores["x2"] = sc2
    
    df_Scores = df_Scores.set_index("Library length")
    
    #df_Scores.plot().get_figure().savefig("/home/ofir/Dropbox/Projects/Peridinium/results/ccm_"+prefix+str(d[x1])+"_"+str(d[x2])+".png")
    df_Scores = df_Scores.fillna(0)
    print(df_Scores)
    Score_X1 = df_Scores["x1"].values[-5:].mean()
    Score_X2 = df_Scores["x2"].values[-5:].mean()
    
    return Score_X1, Score_X2


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
        #tmp_results = []
        #for k in amplified_dfs:
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
    #save envelopes
    
    df_["x1"] = df_["x1"].rolling(5, min_periods=1, center=True).mean()
    df_["x2"] = df_["x2"].rolling(5, min_periods=1, center=True).mean()
    
    
    
    #cwt_peaks_x1 = scipy.signal.find_peaks_cwt(df_["x1"].values.tolist(), widths=np.arange(1, 5))
    #cwt_peaks_x2 = scipy.signal.find_peaks_cwt(df_["x2"].values.tolist(), widths=np.arange(1, 5))
    
    #cwt_peaks_x1 = [p-20 for p in cwt_peaks_x1] 
    #cwt_peaks_x2 = [p-20 for p in cwt_peaks_x2]    
    
    cwt_peaks_x1 = scipy.signal.find_peaks_cwt(df_["x1"].values.tolist(), widths=np.arange(2, 25))
    cwt_peaks_x2 = scipy.signal.find_peaks_cwt(df_["x2"].values.tolist(), widths=np.arange(2, 25))   
      
    cwt_peaks_x1 = [p-20 for p in cwt_peaks_x1] 
    cwt_peaks_x2 = [p-20 for p in cwt_peaks_x2] 
    
    
    cwt_peaks_x1 = [p for p in cwt_peaks_x1 if p <= 0] 
    cwt_peaks_x2 = [p for p in cwt_peaks_x2 if p <= 0] 
    
    max_arg_x1 = 0
    max_arg_x2 = 0
    
    max_arg_x1 = df_['x1'].idxmax()
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





























