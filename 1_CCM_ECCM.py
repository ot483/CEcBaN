
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
    
    #fix outlayers
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
    
    
    #test for stationarity, if not stationary, create diff 
    # perform ADF test and difference if necessary
    def make_stationary(column):
        adf_result = adfuller(column)
        p_value = adf_result[1]
        if p_value >= 0.05:  # If p-value is greater than or equal to 0.05, column is non-stationary
            diff_column = column.diff()  # Difference the column
            return diff_column
        else:
            return column
    
    
        
    for i, vali in enumerate(amplified_dfs):
        vali.columns = [DictCols[i] for i in vali.columns]
        
        # Iterate over columns, perform test, and difference if necessary
        for col in vali.columns:
            try:
                vali[col] = make_stationary(vali[col])
            except:
                print("Error making the dat stationary, most likely the subset consists only a single value.")

        amplified_dfs[i] = vali
        
    
        
        
    #save amplified df as pickle to be read by the external process
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
    
    
        
    
    #check convergence
    
    def check_convergence_conventional(df_Scores):
        try:
            l = int(len(df_Scores) / 4)
            if (df_Scores["x1_mean"][-1*l:].mean() >= df_Scores["x1_mean"][:l].mean()) and \
                    (df_Scores["x1_mean"][-1*l:].mean() >= 0.01) and (df_Scores["x1_mean"][-1*l:].std() <= 0.1):
                return True
            else:
                return False
        except:
            return False
                
    
    
    def check_convergence_density_based(df_Scores, x1, x2):
        X = df_Scores.reset_index()[["Library length", "x1"]].values[int(len(df_Scores)/2):]
    
        def custom_distance_metric(a, b):
            return np.abs(a[1] - b[1])  # Distance based on the y-axis difference
    
        # Apply Agglomerative Clustering with the custom distance metric
        hc = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
        # Create a distance matrix using the custom distance metric
        distance_matrix = np.array([[custom_distance_metric(a, b) for a in X] for b in X])
        labels = hc.fit_predict(distance_matrix)
        unique_labels = set(labels)
    
        #plt.figure(figsize=(10, 6))
        #plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', zorder=1)
        #plt.colorbar(label='Cluster Label')
        #plt.title(str(x1)+" affects "+str(x2))
    
        for k in unique_labels:
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
    
            if xy[:, 1].mean() < 0:
                continue
    
            # Perform linear regression
            model = LinearRegression()
            model.fit(xy[:, 0].reshape(-1, 1), xy[:, 1])
            x_fit = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 100)
            y_fit = model.predict(x_fit.reshape(-1, 1))
    
            #plt.plot(x_fit, y_fit, label=f'Blob {k}', linewidth=2)
            
            if abs(y_fit[-1] - y_fit[0]) <= 0.1:  # TODO add to UI
                return True
    
        return False
        
    # Check convergence
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
    
    # Save the updated results
    with open(outputFolder + 'All_ccm1_results_updated.pickle', 'wb') as handle:
        pickle.dump(All_CCM_dfs, handle)
    
    
    
    # =======
    plt.close()
    
    CausalFeatures  = []
    
    for i in All_CCM_dfs:
        if (len(i[2]) > 0):
            try:
                if (i[1]["x1_mean"][-10:].mean() > 0):# and (i[-1] == True):
                    
                #if (i[-2] == True) and (i[-1] == True):
                    i[1]["x1_mean"].plot()
                    print(i[2][0][2] + ' ' + i[2][0][3])
                    CausalFeatures.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-10:].mean()])
            except:
                    xx=1
    
    df_CausalFeatures = pd.DataFrame(data=CausalFeatures, columns=['x1', 'x2', 'Score'])
    
    df_CausalFeatures = df_CausalFeatures[~df_CausalFeatures["x1"].isin(targetlist)]
    df_CausalFeatures = df_CausalFeatures[~df_CausalFeatures["x2"].isin(confounders)]

    df_CausalFeatures.to_csv(outputFolder+'CCM1_results.csv')

    
    Features = list(df_CausalFeatures['x1'].unique()) + list(df_CausalFeatures['x2'].unique())
    Features = list(set(Features))
    
    #all causal variables vs themselvs
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
    

    #save amplified df as pickle to be read by the external process
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

    # Check convergence
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
                CausalFeatures2.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-10:].mean()])
            except:
                xx=1
    
   
    
    df_CausalFeatures2 = pd.DataFrame(data=CausalFeatures2, columns=['x1', 'x2', 'Score'])
    df_CausalFeatures2["Score"] = df_CausalFeatures2["Score"].round(3)
    
    df_CausalFeatures2.to_csv(outputFolder+'CCM2_results.csv')
    
    df_CausalFeatures2 =  df_CausalFeatures2[(~df_CausalFeatures2['x2'].isin(confounders))]
    df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()
    
    Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
    Features2 = list(set(Features2))
    
  
    #Causal Network
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




def eccm_analysis(df, outputFolder, target_column, confounders):
    """Perform ECCM analysis and return results."""
   
    with open(outputFolder + "AllECCM_results.pickle", 'rb') as handle:
        All_ECCM_results = pickle.load(handle)
   
    print("Performing ECCM analysis...")
    
    # 
    # Filter all ECCM results and extract lags
    df = pd.read_csv(outputFolder+"CCM_ECCM.csv")
    
    for i in All_ECCM_results:
        if i[1] <= 0:
            ind = df[(df['x1'] == i[0]) & (df['x2'] == i[2])]["timeToEffect"].index[0]
            df.loc[ind, "timeToEffect"] = abs(i[1])
            df.loc[ind, "is_Valid"] = 2
    
    # read ccm_eccm_results and create ccm_eccm_results_curated.
    df = df[["x1", "x2", "Score", "is_Valid", "timeToEffect"]]
    
    df = df[~df['x1'].isin([target_column])]
    df = df[~df['x2'].isin(confounders)]
    
    df.to_csv(outputFolder+"CCM_ECCM_curated.csv", index=False)
    
    
    #return print()

    

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

    args = parser.parse_args()

    print("File Path:", args.file_path)
    print("Output Folder:", args.output_folder)
    print("Target Column:", args.target_column)

    df = load_data(args.file_path)
    process_data(df, args.output_folder, args.target_column, args.confounders.split(','), args.subSetLength, args.jumpN, args.z_score_threshold, args.resample_freq,
                 args.embedding_dim, args.lag, args.eccm_window_size, args.number_of_cores, args.file_path, args.ccm_training_proportion, args.max_mi_shift, args.check_convergence)
    
    print("processing data done.")
    
    
    print(" eccm analysis....")
    eccm_analysis(df, args.output_folder, args.target_column,  args.confounders.split(','))

    
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
    
    #df_results = df_results[df_results["Score"] > 0]
    visualize_network(df_results)
    

if __name__ == "__main__":
    main()
