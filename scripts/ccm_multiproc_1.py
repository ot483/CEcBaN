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
import matplotlib.pyplot as plt
import pyEDM
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import skccm as ccm
from skccm.utilities import train_test_split
from scipy.signal import argrelmax, argrelmin




BaseFolder = sys.argv[1]
prefix = sys.argv[2]
cores = int(sys.argv[3])
max_mi_shift = int(sys.argv[4])
embedding_dim = int(sys.argv[5])
lag = int(sys.argv[6])
eccm_window_size = int(sys.argv[7])



def calcCCM(df, x1, x2, prefix, d):

    x_1, x_2 = df[x1], df[x2]  
    try:
    
        #Find optimal E using simplex projection
        libstart = 1
        libend = int(len(df)*0.8)
        predstart = libend+1
        predend = len(df)
        try:
            df_EmbedDimepyEDM = pyEDM.EmbedDimension(dataFrame=df[[x1, x2]].reset_index(),
                          columns = x1,
                          maxE = embedding_dim,#6
                          target = x2,
                          lib = str(libstart)+" "+str(libend),
                          pred = str(predstart)+" "+str(predend),
                          showPlot=False,
                          numThreads=1) 
            
            optimalrho = df_EmbedDimepyEDM["rho"].max()
            embed = df_EmbedDimepyEDM[df_EmbedDimepyEDM["rho"] == optimalrho]["E"].values[0]
            if embed < 3:
                embed = 3
        except:
            embed=5
        ######

        #The second step of CCM is to use the S-map method to test the nonlinearity of the system. In this method, the nonlinear index (θ) is used to govern the weighting procedure, and the nonlinear dynamics system can be identified if the forecast skill improves as θ increases. In Fig. 2(d–f), the nonlinear models (θ > 0) gave better predictions than the linear model (θ = 0), which indicates statistical nonlinear behaviors in these three time series. Therefore, the CCM method can be applied to detect the causality between them.
        try:
            df_PNLpyEDM = pyEDM.PredictNonlinear(dataFrame = df[[x1,x2]].reset_index(),
                          E = int(embed),
                          columns = x1,
                          lib = str(libstart)+" "+str(libend),
                          pred = str(predstart)+" "+str(predend),
                          showPlot = False) 
            
            if (df_PNLpyEDM["rho"].max() != df_PNLpyEDM["rho"].values.tolist()[0]): \
                #and (df_PNLpyEDM["rho"].max() == df_PNLpyEDM["rho"].values.tolist()[-1]):
                    NonLinearity = True
            else:
                   NonLinearity =False
                   return [0, 0, 0, 0, False, 0, 0] 
        except:
            NonLinearity =False
        #####
        #NonLinearity = True###
        #TODO - replace skccm to pyEDM
        
        e1 = ccm.Embed(x_1)
        e2 = ccm.Embed(x_2)
        
        #Find optimal lag using mutual information.
        #lagX1 = 2
        #if lag == 0:
        arr = e1.mutual_information(max_mi_shift)
        arr = pd.DataFrame(arr).ewm(span = 3).mean().values
        #pd.DataFrame(arr).plot()
        try:
            lagX1 = int(argrelmin(arr)[0][0])
        except:
            lagX1 = 2
        #print(arr[idx][0])
        
        if lagX1 < 2:
            lagX1 = 2        
        
        if lagX1 > lag:
            lagX1 = lag
        
        lagX2 = lagX1    
        
        lagX1 = int(lagX1)
        lagX2 = int(lagX2)
        embed = int(embed)        
        
        print("Selected lag "+str(lagX1))
        print("Selected embedding dim "+str(embed))


        X1 = e1.embed_vectors_1d(lagX1,embed)
        X2 = e2.embed_vectors_1d(lagX2,embed)
        
        
        #split the embedded time series
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)
        
        CCM = ccm.CCM() #initiate the class
        
        #library lengths to test
        len_tr = len(x1tr)
        print("len_tr "+str(len_tr))
        #lib_lens = np.arange(10, len_tr, len_tr/2, dtype='int')
        lib_lens = list(range(5, len_tr-1, 1))
        
        #test causation
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
        
        sc1,sc2 = CCM.score()
        
        df_Scores = pd.DataFrame()
        df_Scores["Library length"] = lib_lens
        df_Scores["x1"] = sc1
        df_Scores["x2"] = sc2
        
        df_Scores = df_Scores.set_index("Library length")
        #fig, axs = plt.subplots(figsize=(10, 10))
        
        #df_Scores.plot().get_figure().savefig(BaseFolder+"ccm_"+prefix+str(x1)+"_"+str(x2)+".png")
        #fig.savefig(BaseFolder+"ccm_"+prefix+str(x1)+"_"+str(x2)+".png")
        #plt.close()
        
        df_Scores = df_Scores.fillna(0)
        Score_X1 = df_Scores["x1"].values[-5:].mean()

    #pd.DataFrame(convStd).plot()
    
    except Exception as e: 
        print(e)   
        lagX1 = 2
        embed = 5
        df_Scores, Score_X1, x1, x2, NonLinearity = 0, 0, 0, 0, False
    
    if (x1 in list(d.keys())) and (x2 in list(d.keys())):
        return [df_Scores, Score_X1, d[x1], d[x2], NonLinearity, lagX1, embed]
    else:
        return [0, 0, 0, 0, False, 0, 0]




def fullCCM(dfsList, col, targetCol, dic, prefix_, showFig = False):
        tmp_results = []
        for j in dfsList:
    
            j = j.fillna(0)
            tmp_results.append(calcCCM(j,
                                      x1=dic[col],
                                      x2=dic[targetCol],
                                      prefix=prefix_,
                                      d = dic.copy())) 
        #print(str(len(ChemCols))+"/"+str(count))
        #here collect and store all x1's
        Final_results = []
        calculated_dfs = []
        for k, valk in enumerate(tmp_results):
            if valk[-3] == True:
                calculated_dfs.append( valk[0].reset_index())
                Final_results.append(valk)
        
        if len(calculated_dfs) > 1:        
            c = pd.concat(calculated_dfs, axis=0, ignore_index=False)
            c_means = pd.DataFrame()
            c_means["x1_mean"] = c.groupby("Library length")["x1"].agg("mean") 
            c_means["x2_mean"] = c.groupby("Library length")["x2"].agg("mean") 
            c_means = c_means.reset_index()
            
            
            if showFig == True:
                # Calculate the point density
                 try:
# =============================================================================
#                     xy = np.vstack([ c["Library length"].values, c["x1"].values])
#                     z = gaussian_kde(xy)(xy)
#                     
#                     fig, ax = plt.subplots()
#                     ax.scatter(c["Library length"].values,  c["x1"].values, c=z, s=1)
#                     ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="red", s=7)
#                     ax.scatter(c_means["Library length"].values, c_means["x2_mean"].values, color="gray", s=7)
#             
#                     plt.savefig(BaseFolder+"ccm_"+col+"_"+targetCol+".png")
#                     plt.close()
#                      
# =============================================================================
                                         
                                    
                    # Calculate the density
                    xy = np.vstack([c["Library length"].values, c["x1"].values])
                    kde = gaussian_kde(xy)
                    
                    # Create grid for density plot
                    x_grid = np.linspace(c["Library length"].min(), c["Library length"].max(), 100)
                    y_grid = np.linspace(c["x1"].min(), c["x1"].max(), 100)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(kde(positions).T, X.shape)
                    
                    # Plotting
                    fig, ax = plt.subplots()
                    
                    # Density plot
                    cax = ax.imshow(Z, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                                    origin='lower', cmap='coolwarm', aspect='auto')
                    
                    # Contour plot
                    ax.contour(X, Y, Z, colors='blue')
                    
                    # Mean points
                    ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="black", s=7)
                    #ax.scatter(c_means["Library length"].values, c_means["x2_mean"].values, color="gray", s=7)
                    
                    # Adding colorbar
                    cbar = fig.colorbar(cax, ax=ax)
                    cbar.set_label('Density')
                    ax.set_title('Density Plot of ' + col + ' affects ' + targetCol)
                    ax.set_xlabel('Library size (l)')
                    ax.set_ylabel('rho (p)') 

                    plt.savefig(BaseFolder+"ccm_density_"+col+"_"+targetCol+".png")
                    plt.close()
                    
                 except:
                    fig, ax = plt.subplots()
                    ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="red", s=3)
                    #ax.scatter(c_means["Library length"].values, c_means["x2_mean"].values, color="gray", s=3)
                    ax.set_xlabel('Library size (l)')
                    ax.set_ylabel('rho (p)') 
                    plt.savefig(BaseFolder+"ccm_"+col+"_"+targetCol+".png")               
                    plt.close()
                
            print(c_means)        
            return c, c_means, Final_results   
        else:
            return 0, 0, Final_results
   




# =============================================================================
# 
# BaseFolder = sys.argv[1]
# prefix = sys.argv[2]
# cores = int(sys.argv[3])
# 
# =============================================================================
with open(BaseFolder + prefix + 'amplified_dfs.pickle', 'rb') as handle:
    amplified_dfs = pickle.load(handle)

with open(BaseFolder + prefix + 'DictCols.pickle', 'rb') as handle:
    DictCols = pickle.load(handle)
    
with open(BaseFolder + prefix +'x1_x2_columns.pickle', 'rb') as handle:
    cols_x1, cols_x2 = pickle.load(handle)

#dont show figs if its used for surrogates
if "surr_" in prefix:
    sf = False
else:
    sf = True

def manipulate(v):
    All_causal_CCM_dfs = []
    for j, valj in enumerate(cols_x2):
            All_causal_CCM_dfs.append(fullCCM(dfsList=amplified_dfs,
                        col=v,
                        targetCol=valj,
                        dic=DictCols,
                        prefix_=prefix,
                        #lag=0,
                        showFig = sf))
            
            print(str(valj)+" : "+str(v))    
    return All_causal_CCM_dfs    




pool = Pool(cores)   

results_list_final = []
results_list_final = pool.map(manipulate,cols_x1)

results_list_fixed = []
for i in results_list_final:
    results_list_fixed = results_list_fixed + i

pool.close()
pool.join()
print('end')
    
with open(BaseFolder + 'All_' + prefix + 'results.pickle', 'wb') as handle:
    pickle.dump(results_list_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)      
    
    





























