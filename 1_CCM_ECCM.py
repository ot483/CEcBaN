
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import networkx as nx
import os
from statsmodels.tsa.stattools import adfuller
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering



BaseFolder = "./"



def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def process_data(df, output_folder, target_column, confounders, subSetLength, jumpN, z_score_threshold, resample_freq, embedding_dim,
                 lag, eccm_window_size, number_of_cores, file_path, ccm_training_proportion, max_mi_shift, check_convergence):
    """Process the data based on given parameters."""
    print("Processing data...")
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
        
    
    outputFolder = output_folder
    targetlist = [target_column]
    
   
    
    concated_ = df
    try:
        concated_['Date'] = pd.to_datetime(concated_['Date'])
        concated_ = concated_.set_index('Date')
        date_col = 'Date'
    except:
        concated_['date'] = pd.to_datetime(concated_['date'])
        concated_ = concated_.set_index('date')        
        date_col = 'date'
    
    Full_cols = [i for i in concated_.columns if not i in ["Date", "index", "date"]]
  
    
    df_interpolatedNotNormalized = concated_.copy()
    
    df_upsampled_normalized = pd.DataFrame(index = concated_.index)
    AllScalersDict = {}
    for i in concated_.columns:
        scaler = MinMaxScaler((0,1))
        scaled_data = scaler.fit_transform(concated_[i].values.reshape(-1, 1))
        df_upsampled_normalized[i] = [j[0] for j in scaled_data]
        AllScalersDict[i] = scaler
    
    df_concated_fixed_outlayers = df_upsampled_normalized.copy()
    
    for i in df_concated_fixed_outlayers.columns:
        mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > z_score_threshold)
        df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate()
    
    for i in df_interpolatedNotNormalized.columns:
        mask = (np.abs(stats.zscore(df_interpolatedNotNormalized[i])) > z_score_threshold)
        df_interpolatedNotNormalized[i] = df_interpolatedNotNormalized[i].mask(mask).interpolate(method='linear')
        
    df_interpolatedNotNormalized = df_interpolatedNotNormalized.resample(resample_freq).interpolate(method='linear') #its already 7 days, this interpolation is for the case there are missing values
    df_interpolatedNotNormalized[df_interpolatedNotNormalized < 0] = 0
    
    df_concated_fixed_outlayers = df_concated_fixed_outlayers.dropna()
    

    Full_cols  = list(set(Full_cols + targetlist))
    
    ###############################################
    
    df_upsampled_proc = df_concated_fixed_outlayers.copy()

    amplified_dfs = amplifyData(df_upsampled_proc, subSetLength=subSetLength, jumpN=jumpN)
                    
    DictCols = build_colsDict(df_upsampled_proc)
    
    
    def make_stationary(column):
        adf_result = adfuller(column)
        p_value = adf_result[1]
        if p_value >= 0.05:  
            diff_column = column.diff()  
            return diff_column
        else:
            return column
         
    for i, vali in enumerate(amplified_dfs):
        vali.columns = [DictCols[i] for i in vali.columns]
        
        for col in vali.columns:
            try:
                vali[col] = make_stationary(vali[col])
            except:
                print("Error making the dat stationary, most likely the subset consists only a single value.")

        amplified_dfs[i] = vali
        
        
    with open(outputFolder+'ccm1_amplified_dfs.pickle', 'wb') as handle:
        pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    with open(outputFolder+'ccm1_DictCols.pickle', 'wb') as handle:
        pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
    with open(outputFolder+'ccm1_x1_x2_columns.pickle', 'wb') as handle:
        pickle.dump([Full_cols, targetlist], handle, protocol=pickle.HIGHEST_PROTOCOL)     
    
    os.system('python '+BaseFolder+'/scripts/ccm_multiproc_1.py '+ outputFolder + ' ccm1_ ' + str(number_of_cores) +\
                                                                                          " " + str(max_mi_shift) +\
                                                                                          " " + str(embedding_dim) +\
                                                                                          " " + str(lag) +\
                                                                                          " " + str(eccm_window_size))    
    
    with open(outputFolder + 'All_ccm1_results.pickle', 'rb') as handle:
        All_CCM_dfs = pickle.load(handle)
    
    
    def check_convergence_conventional(df_Scores):
        try:
            l = int(len(df_Scores))
            if (df_Scores["x1_mean"][:int(0.75*l)].mean() <= df_Scores["x1_mean"][int(0.75*l):int(l*0.95)].mean()) and \
             (df_Scores["x1_mean"][int(0.75*l):].mean() >= 0) and (df_Scores["x1_mean"][int(0.7*l):int(0.9*l)].std() <= 0.15):
                return True
            else:
                return False
        except:
            return False
                
    
    
    def check_convergence_density_based(df_Scores):
        """
        Strict convergence detection - only plateau or increasing trends allowed.
        Much more stringent than original.
        """
        try:
            if df_Scores is None or len(df_Scores) < 15:  # Need more data
                return False
            
            if "x1_mean" not in df_Scores.columns:
                return False
            
            SKILL_THRESHOLD = 0.1      
            VARIANCE_THRESHOLD = 0.1   
            MIN_IMPROVEMENT = 0.02     
            RECENT_SLOPE_THRESHOLD = -0.1  
            TAIL_POINTS = int(len(df_Scores)*0.5)            
            
            l = max(3, int(len(df_Scores) / 3))
            
            skill_data = df_Scores["x1_mean"].dropna()
            if len(skill_data) < 10:
                return False
            
            early_vals = skill_data[:l]
            recent_vals = skill_data[-1*l:]
            
            early_mean = early_vals.mean()
            recent_mean = recent_vals.mean()
            recent_std = recent_vals.std()
            
            skill_sufficient = recent_mean >= SKILL_THRESHOLD            
            skill_improving = (recent_mean - early_mean) >= MIN_IMPROVEMENT      
            variance_stable = recent_std <= VARIANCE_THRESHOLD
            
            if len(recent_vals) >= 5:
                x_vals = np.arange(len(recent_vals))
                recent_slope = np.polyfit(x_vals, recent_vals.values, 1)[0]
                trend_stable = recent_slope >= RECENT_SLOPE_THRESHOLD
            else:
                trend_stable = False
            
            if len(skill_data) >= TAIL_POINTS:
                tail_vals = skill_data[-TAIL_POINTS:].values
                tail_slope = np.polyfit(range(TAIL_POINTS), tail_vals, 1)[0]
                no_tail_decline = tail_slope >= -0.1  
            else:
                no_tail_decline = True
            
            top_30_threshold = np.percentile(skill_data.values, 70)
            recent_performance_good = recent_mean >= top_30_threshold
            
            converged = (skill_sufficient and 
                        skill_improving and 
                        variance_stable and 
                        #trend_stable and 
                        no_tail_decline)# and
                        #recent_performance_good)
            
            return converged
            
        except Exception as e:
            print(f"Convergence check error: {e}")
            return False
        
    for counti, i in enumerate(All_CCM_dfs):
        All_CCM_dfs[counti] = list(All_CCM_dfs[counti])
        
        if check_convergence == "density":
            try:
                print(i[1])
                df_Scores = i[1]
                convergence = check_convergence_density_based(df_Scores)
            except:
                convergence = False
        else:
            df_Scores = i[1]
            convergence = check_convergence_conventional(df_Scores)
        
        All_CCM_dfs[counti].append(convergence)
        if convergence:
            print('true')
    
    with open(outputFolder + 'All_ccm1_results_updated.pickle', 'wb') as handle:
        pickle.dump(All_CCM_dfs, handle)
    
    
    
    # =======
    plt.close()
    
    CausalFeatures  = []
    
    for i in All_CCM_dfs:
        if (len(i[2]) > 0):
            try:
                    
                if (i[1]["x1_mean"][-20:-10].mean() > 0) and (i[-1] == True):
                    i[1]["x1_mean"].plot()
                    print(i[2][0][2] + ' ' + i[2][0][3])
                    CausalFeatures.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-20:-10].mean()])
            except:
                    xx=1
    
    df_CausalFeatures = pd.DataFrame(data=CausalFeatures, columns=['x1', 'x2', 'Score'])
    
    df_CausalFeatures = df_CausalFeatures[~df_CausalFeatures["x1"].isin(targetlist)]
    df_CausalFeatures = df_CausalFeatures[~df_CausalFeatures["x2"].isin(confounders)]

    df_CausalFeatures.to_csv(outputFolder+'CCM1_results.csv')

    
    Features = list(df_CausalFeatures['x1'].unique()) + list(df_CausalFeatures['x2'].unique())
    Features = list(set(Features))
    
    Features = Features + targetlist
    Features = list(set(Features))
    Features = [i for i in Features if i in list(concated_.columns)]
    
    amplified_dfs = amplifyData(df_upsampled_proc[Features], subSetLength=subSetLength, jumpN=jumpN)
    
    
    DictCols = {}
    DictCols = build_colsDict(df_upsampled_proc[Features])
    
    
    
    for i, vali in enumerate(amplified_dfs):
        vali.columns = [DictCols[i] for i in vali.columns]
        
        for col in vali.columns:
            try:
                vali[col] = make_stationary(vali[col])
            except:
                print("Error making the dat stationary, most likely the subset consists only a single value.")
        amplified_dfs[i] = vali
    
    Features2 = [i for i in Features if not i in confounders]
    Features = [i for i in Features if not i in targetlist]

    with open(outputFolder+'ccm2_amplified_dfs.pickle', 'wb') as handle:
        pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    with open(outputFolder+'ccm2_DictCols.pickle', 'wb') as handle:
        pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
    with open(outputFolder+'ccm2_x1_x2_columns.pickle', 'wb') as handle:
        pickle.dump([Features, Features2], handle, protocol=pickle.HIGHEST_PROTOCOL)     
    
    os.system('python '+BaseFolder+'/scripts/ccm_multiproc_1.py '+ outputFolder + ' ccm2_ ' + str(number_of_cores) +\
                                                                                          " " + str(max_mi_shift) +\
                                                                                          " " + str(embedding_dim) +\
                                                                                          " " + str(lag) +\
                                                                                          " " + str(eccm_window_size))    
    
    #===========
    with open(outputFolder + 'All_ccm2_results.pickle', 'rb') as handle:
        All_causal_CCM_dfs = pickle.load(handle)
    
    
    x=0

    for counti, i in enumerate(All_CCM_dfs):
        All_CCM_dfs[counti] = list(All_CCM_dfs[counti])
        
        if check_convergence == "density":
            try:
                df_Scores = pd.concat([j[0] for j in i[2]])
                convergence = check_convergence_density_based(df_Scores, i[2][0][2], i[2][0][3])
            except:
                convergence = False
        else:
            df_Scores = i[1]
            convergence = check_convergence_conventional(df_Scores)
        
        All_CCM_dfs[counti].append(convergence)
        if convergence:
            print('true')
    
    CausalFeatures2  = []

    
    for i in All_causal_CCM_dfs:
        if (len(i[2]) > 0):
            
            try:
                CausalFeatures2.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][int(0.66*len(i[1])):].mean()])
            except:
                xx=1
    
   
    df_CausalFeatures2 = pd.DataFrame(data=CausalFeatures2, columns=['x1', 'x2', 'Score'])
  
    if "Score" in df_CausalFeatures2.columns:
        try:
            df_CausalFeatures2["Score"] = pd.to_numeric(df_CausalFeatures2["Score"], errors="coerce")
            df_CausalFeatures2 = df_CausalFeatures2.dropna(subset=["Score"])
        except:
            pass
    if "Score" in df_CausalFeatures2.columns:
        try:
            df_CausalFeatures2["Score"] = pd.to_numeric(df_CausalFeatures2["Score"], errors="coerce")
            df_CausalFeatures2 = df_CausalFeatures2.dropna(subset=["Score"])
        except:
            pass
    df_CausalFeatures2["Score"] = df_CausalFeatures2["Score"].round(3)
    
    df_CausalFeatures2.to_csv(outputFolder+'CCM2_results.csv')
    
    df_CausalFeatures2 =  df_CausalFeatures2[(~df_CausalFeatures2['x2'].isin(confounders))]
    df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()
    
    Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
    Features2 = list(set(Features2))
    
    G = nx.DiGraph() 
    
    for i in df_CausalFeatures2[["x1", "x2", "Score"]].values.tolist():
        G.add_edge(i[0], i[1], weight = abs(i[2])*10)
    
    df_CausalFeatures2 = df_CausalFeatures2.assign(is_Valid=[np.nan]*len(df_CausalFeatures2))
    df_CausalFeatures2 = df_CausalFeatures2.assign(timeToEffect=[np.nan]*len(df_CausalFeatures2))
    df_CausalFeatures2 = df_CausalFeatures2.reset_index(drop=True)
    df_CausalFeatures2.to_csv(outputFolder+'CCM_ECCM.csv')
    
    
    #ECCM ###############################################
    df_CausalFeatures2 = pd.read_csv(outputFolder+'CCM_ECCM.csv')
    
    df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] > 0] 
    
    df_upsampled_proc = df_concated_fixed_outlayers.dropna().copy()
        
    x1x2s = df_CausalFeatures2[['x1', 'x2']].values.tolist()
    x1x2s = [(i[0], i[1]) for i in x1x2s]
    
    with open(outputFolder+'eccm_dataset.pickle', 'wb') as handle:
        pickle.dump(df_upsampled_proc, handle, protocol=pickle.HIGHEST_PROTOCOL)      
       
    with open(outputFolder+'eccm_edges.pickle', 'wb') as handle:
        pickle.dump(x1x2s, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    os.system('python '+BaseFolder+'scripts/eccm_multiproc_1.py ' + outputFolder +' '+ str(number_of_cores) +\
                                                                                              " " + str(max_mi_shift) +\
                                                                                              " " + str(embedding_dim) +\
                                                                                                " " + str(lag) +\
                                                                                                " " + str(eccm_window_size))


def eccm_analysis(df, outputFolder, target_column, confounders, prefer_zero_lag='true'):
    """Perform ECCM analysis and return results."""
   
    with open(outputFolder + "AllECCM_results.pickle", 'rb') as handle:
        All_ECCM_results = pickle.load(handle)
   
    print("Performing ECCM analysis...")
    
    df = pd.read_csv(outputFolder+"CCM_ECCM.csv")
    
    for i in All_ECCM_results:
        if i[1] <= 0:
            ind = df[(df['x1'] == i[0]) & (df['x2'] == i[2])]["timeToEffect"].index[0]
            
            if prefer_zero_lag == 'true':
                df.loc[ind, "timeToEffect"] = 0
                df.loc[ind, "is_Valid"] = 2
            else:
                df.loc[ind, "timeToEffect"] = abs(i[1])
                df.loc[ind, "is_Valid"] = 2                
    
    df = df[["x1", "x2", "Score", "is_Valid", "timeToEffect"]]
    
    df = df[~df['x1'].isin([target_column])]
    df = df[~df['x2'].isin(confounders)]
    
    df.to_csv(outputFolder+"CCM_ECCM_curated.csv", index=False)
    
    

def main():
    parser = argparse.ArgumentParser(description="Process data for CCM and ECCM analysis.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to unique output folder for each run')
    parser.add_argument('--target_column', type=str, required=True, help='Target column for analysis')
    parser.add_argument('--confounders', type=str, default="", help='Comma-separated list of confounder columns')
    parser.add_argument('--subSetLength', type=int, default=60, help='Subset length for analysis')
    parser.add_argument('--jumpN', type=int, default=30, help='Jump N value for processing')
    parser.add_argument('--z_score_threshold', type=float, default=3.0, help='Z-score threshold for outlier detection')
    parser.add_argument('--resample_freq', type=str, default='D', help='Frequency for data resampling')
    parser.add_argument('--embedding_dim', type=int, default=2, help='Embedding dimension for CCM')
    parser.add_argument('--lag', type=int, default=1, help='Lag for CCM')
    parser.add_argument('--eccm_window_size', type=int, default=50, help='Window size for ECCM')
    parser.add_argument('--number_of_cores', type=int, default=1, help='Number of cores for multithreading')
    parser.add_argument('--ccm_training_proportion', type=float, default=0.75, help='CCM training proportion in CCM calculation')
    parser.add_argument('--max_mi_shift', type=int, default=20, help='Max mutual information shift')
    parser.add_argument('--check_convergence', type=str, default='density', help='choose convergence indetification method')
    parser.add_argument('--prefer_zero_lag', type=str, default='true', help='Prefer immediate effects (lag 0) when meaningful, or always use strongest effect')


    args = parser.parse_args()

    print("File Path:", args.file_path)
    print("Output Folder:", args.output_folder)
    print("Target Column:", args.target_column)

    df = load_data(args.file_path)
    process_data(df, args.output_folder, args.target_column, args.confounders.split(','), args.subSetLength, args.jumpN, args.z_score_threshold, args.resample_freq,
                 args.embedding_dim, args.lag, args.eccm_window_size, args.number_of_cores, args.file_path, args.ccm_training_proportion, args.max_mi_shift, args.check_convergence)
    
    print("processing data done.")
    print(" eccm analysis....")
    eccm_analysis(df, args.output_folder, args.target_column,  args.confounders.split(','), args.prefer_zero_lag)

    
    df_results = pd.read_csv(args.output_folder+"CCM_ECCM_curated.csv")
    
    def visualize_network(results, filepath=args.output_folder+"network_plot.png"):
        """Visualize the results of CCM ECCM analysis as a network graph and save as an image."""
        G = nx.DiGraph()
        results = results[results["Score"] > 0]
        results = results[results["is_Valid"] == 2]

        for source, target, weight in results[["x1", "x2", "Score"]].values.tolist():
            G.add_edge(source, target, weight=weight)
    
        pos = nx.spring_layout(G)
        weights = nx.get_edge_attributes(G, 'weight')
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', width=3, edge_cmap=plt.cm.Blues)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
        plt.title('CCM-ECCM Network')
        plt.savefig(filepath)
        plt.close()
    
    visualize_network(df_results)
    
if __name__ == "__main__":
    main()
