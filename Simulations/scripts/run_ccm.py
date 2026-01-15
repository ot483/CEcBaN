
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
                 lag, eccm_window_size, number_of_cores, file_path, ccm_training_proportion, max_mi_shift, check_convergence,
                 conv_skill_threshold=0.1, conv_variance_threshold=0.1, conv_min_improvement=0.02):
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

        """
        try:
            if df_Scores is None or len(df_Scores) < 15:  # Need more data
                return False
            
            if "x1_mean" not in df_Scores.columns:
                return False
            
            SKILL_THRESHOLD = conv_skill_threshold
            VARIANCE_THRESHOLD = conv_variance_threshold
            MIN_IMPROVEMENT = conv_min_improvement
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


    def calculate_convergence_confidence(df_Scores):
        """
         Quantitative convergence confidence scoring (0-1 scale)

        Evaluates convergence quality based on four components (25% each):
        1. Positive skill: Final prediction skill > 0
        2. Increasing trend: Skill increases from early to late
        3. Stability: Variance in late-stage skill is low (< 0.15)
        4. Monotonic improvement: Skill increases across early, middle, late periods

        Returns:
            confidence_score (float): 0-1, where:
                > 0.7 = high confidence convergence
                0.4-0.7 = medium confidence (manual review recommended)
                < 0.4 = low confidence non-convergence
            components (dict): Breakdown of the four scoring components
        """
        try:
            if df_Scores is None or len(df_Scores) < 10:
                return 0.0, {'positive_skill': 0, 'increasing_trend': 0,
                            'stability': 0, 'monotonic': 0}

            if "x1_mean" not in df_Scores.columns:
                return 0.0, {'positive_skill': 0, 'increasing_trend': 0,
                            'stability': 0, 'monotonic': 0}

            skill_data = df_Scores["x1_mean"].dropna()
            if len(skill_data) < 10:
                return 0.0, {'positive_skill': 0, 'increasing_trend': 0,
                            'stability': 0, 'monotonic': 0}

            # Divide into early, middle, late periods
            n = len(skill_data)
            early_end = n // 3
            middle_end = 2 * n // 3

            early_vals = skill_data[:early_end]
            middle_vals = skill_data[early_end:middle_end]
            late_vals = skill_data[middle_end:]

            early_mean = early_vals.mean()
            middle_mean = middle_vals.mean()
            late_mean = late_vals.mean()
            late_std = late_vals.std()

            # Component 1: Positive skill (25%)
            # Final skill should be > 0 (better than random)
            if late_mean > 0.2:
                positive_skill_score = 1.0
            elif late_mean > 0:
                positive_skill_score = late_mean / 0.2  # Linear scaling 0-0.2
            else:
                positive_skill_score = 0.0

            # Component 2: Increasing trend (25%)
            # Skill should increase from early to late
            improvement = late_mean - early_mean
            if improvement > 0.1:
                increasing_trend_score = 1.0
            elif improvement > 0:
                increasing_trend_score = improvement / 0.1  # Linear scaling 0-0.1
            else:
                increasing_trend_score = 0.0

            # Component 3: Stability (25%)
            # Late-stage variance should be low (< 0.15)
            if late_std < 0.05:
                stability_score = 1.0
            elif late_std < 0.15:
                stability_score = 1.0 - (late_std - 0.05) / 0.1  # Linear scaling 0.05-0.15
            else:
                stability_score = 0.0

            # Component 4: Monotonic improvement (25%)
            # Skill should increase from early -> middle -> late
            early_to_middle = middle_mean >= early_mean
            middle_to_late = late_mean >= middle_mean

            if early_to_middle and middle_to_late:
                monotonic_score = 1.0
            elif early_to_middle or middle_to_late:
                monotonic_score = 0.5
            else:
                monotonic_score = 0.0

            # Total confidence score (equal weighting: 25% each)
            confidence_score = 0.25 * (positive_skill_score +
                                      increasing_trend_score +
                                      stability_score +
                                      monotonic_score)

            components = {
                'positive_skill': positive_skill_score,
                'increasing_trend': increasing_trend_score,
                'stability': stability_score,
                'monotonic': monotonic_score,
                'early_mean': early_mean,
                'middle_mean': middle_mean,
                'late_mean': late_mean,
                'late_std': late_std,
                'improvement': improvement
            }

            return confidence_score, components

        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.0, {'positive_skill': 0, 'increasing_trend': 0,
                        'stability': 0, 'monotonic': 0}


    # Track convergence confidence scores
    convergence_confidence_data = []

    for counti, i in enumerate(All_CCM_dfs):
        All_CCM_dfs[counti] = list(All_CCM_dfs[counti])

        if check_convergence == "density":
            try:
                print(i[1])
                df_Scores = i[1]
                convergence = check_convergence_density_based(df_Scores)
            except:
                convergence = False
                df_Scores = None
        else:
            df_Scores = i[1]
            convergence = check_convergence_conventional(df_Scores)

        # Calculate convergence confidence score
        if df_Scores is not None:
            confidence_score, components = calculate_convergence_confidence(df_Scores)
        else:
            confidence_score = 0.0
            components = {'positive_skill': 0, 'increasing_trend': 0,
                         'stability': 0, 'monotonic': 0,
                         'early_mean': 0, 'middle_mean': 0,
                         'late_mean': 0, 'late_std': 0, 'improvement': 0}

        # Store confidence data
        try:
            x1 = i[2][0][2] if len(i[2]) > 0 else 'unknown'
            x2 = i[2][0][3] if len(i[2]) > 0 else 'unknown'
        except:
            x1 = 'unknown'
            x2 = 'unknown'

        convergence_confidence_data.append({
            'x1': x1,
            'x2': x2,
            'convergence_binary': convergence,
            'confidence_score': confidence_score,
            'confidence_category': ('high' if confidence_score > 0.7
                                   else 'medium' if confidence_score > 0.4
                                   else 'low'),
            'positive_skill': components['positive_skill'],
            'increasing_trend': components['increasing_trend'],
            'stability': components['stability'],
            'monotonic': components['monotonic'],
            'early_mean': components.get('early_mean', 0),
            'middle_mean': components.get('middle_mean', 0),
            'late_mean': components.get('late_mean', 0),
            'late_std': components.get('late_std', 0),
            'improvement': components.get('improvement', 0)
        })

        All_CCM_dfs[counti].append(convergence)
        if convergence:
            print('true')
    
    with open(outputFolder + 'All_ccm1_results_updated.pickle', 'wb') as handle:
        pickle.dump(All_CCM_dfs, handle)

    # Save convergence confidence scores
    print("\n" + "="*70)
    print("CONVERGENCE CONFIDENCE SUMMARY")
    print("="*70)

    if len(convergence_confidence_data) > 0:
        df_confidence = pd.DataFrame(convergence_confidence_data)

        # Sort by confidence score (descending)
        df_confidence = df_confidence.sort_values('confidence_score', ascending=False)

        # Save to CSV
        try:
            csv_path = outputFolder + 'convergence_confidence_summary.csv'
            df_confidence.to_csv(csv_path, index=False)
            print(f"\n✓ Saved convergence confidence scores to: {csv_path}")
        except Exception as e:
            print(f"\n!  Warning: Could not save confidence CSV: {e}")

        # Calculate statistics
        n_total = len(df_confidence)
        n_high = (df_confidence['confidence_category'] == 'high').sum()
        n_medium = (df_confidence['confidence_category'] == 'medium').sum()
        n_low = (df_confidence['confidence_category'] == 'low').sum()
        n_converged_binary = df_confidence['convergence_binary'].sum()

        print(f"\nTotal variable pairs analyzed: {n_total}")
        print(f"Converged (binary check): {n_converged_binary}")
        print(f"\nConfidence Distribution:")
        print(f"  High confidence (>0.7):    {n_high:3d} ({n_high/n_total*100:5.1f}%)")
        print(f"  Medium confidence (0.4-0.7): {n_medium:3d} ({n_medium/n_total*100:5.1f}%)")
        print(f"  Low confidence (<0.4):     {n_low:3d} ({n_low/n_total*100:5.1f}%)")

        # Show top 5 highest confidence pairs
        if len(df_confidence) > 0:
            print("\nTop 5 highest confidence convergence pairs:")
            print("-" * 70)
            top5 = df_confidence.head(5)
            for idx, row in top5.iterrows():
                cat_marker = "✓✓" if row['confidence_category'] == 'high' else "✓" if row['confidence_category'] == 'medium' else "○"
                print(f"  {cat_marker} {row['x1']:20s} → {row['x2']:20s} | "
                      f"score={row['confidence_score']:.3f}, late_skill={row['late_mean']:.3f}")

        # Identify borderline cases (medium confidence) for manual review
        medium_conf = df_confidence[df_confidence['confidence_category'] == 'medium']
        if len(medium_conf) > 0:
            print(f"\n!  {len(medium_conf)} pairs have medium confidence - consider manual review:")
            print("-" * 70)
            for idx, row in medium_conf.head(10).iterrows():
                print(f"    {row['x1']:20s} → {row['x2']:20s} | score={row['confidence_score']:.3f}")
            if len(medium_conf) > 10:
                print(f"    ... and {len(medium_conf) - 10} more (see CSV for full list)")

        print("="*70)
    else:
        print("\n!  Warning: No convergence confidence data to save")



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

    # Track convergence confidence scores for CCM2/ECCM
    convergence_confidence_data2 = []

    for counti, i in enumerate(All_CCM_dfs):
        All_CCM_dfs[counti] = list(All_CCM_dfs[counti])

        if check_convergence == "density":
            try:
                df_Scores = pd.concat([j[0] for j in i[2]])
                convergence = check_convergence_density_based(df_Scores, i[2][0][2], i[2][0][3])
            except:
                convergence = False
                df_Scores = None
        else:
            df_Scores = i[1]
            convergence = check_convergence_conventional(df_Scores)

        #  Calculate convergence confidence score for CCM2/ECCM
        if df_Scores is not None:
            confidence_score, components = calculate_convergence_confidence(df_Scores)
        else:
            confidence_score = 0.0
            components = {'positive_skill': 0, 'increasing_trend': 0,
                         'stability': 0, 'monotonic': 0,
                         'early_mean': 0, 'middle_mean': 0,
                         'late_mean': 0, 'late_std': 0, 'improvement': 0}

        # Store confidence data
        try:
            x1 = i[2][0][2] if len(i[2]) > 0 else 'unknown'
            x2 = i[2][0][3] if len(i[2]) > 0 else 'unknown'
        except:
            x1 = 'unknown'
            x2 = 'unknown'

        convergence_confidence_data2.append({
            'x1': x1,
            'x2': x2,
            'convergence_binary': convergence,
            'confidence_score': confidence_score,
            'confidence_category': ('high' if confidence_score > 0.7
                                   else 'medium' if confidence_score > 0.4
                                   else 'low'),
            'positive_skill': components['positive_skill'],
            'increasing_trend': components['increasing_trend'],
            'stability': components['stability'],
            'monotonic': components['monotonic'],
            'early_mean': components.get('early_mean', 0),
            'middle_mean': components.get('middle_mean', 0),
            'late_mean': components.get('late_mean', 0),
            'late_std': components.get('late_std', 0),
            'improvement': components.get('improvement', 0)
        })

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

    if "Score" in df_CausalFeatures2.columns:
        try:
            df_CausalFeatures2["Score"] = pd.to_numeric(df_CausalFeatures2["Score"], errors="coerce")
            df_CausalFeatures2 = df_CausalFeatures2.dropna(subset=["Score"])
        except:
            pass
    df_CausalFeatures2["Score"] = df_CausalFeatures2["Score"].round(3)
    
    df_CausalFeatures2.to_csv(outputFolder+'CCM2_results.csv')

    #  Save convergence confidence scores for CCM2/ECCM
    print("\n" + "="*70)
    print("ECCM CONVERGENCE CONFIDENCE SUMMARY")
    print("="*70)

    if len(convergence_confidence_data2) > 0:
        df_confidence2 = pd.DataFrame(convergence_confidence_data2)

        # Sort by confidence score (descending)
        df_confidence2 = df_confidence2.sort_values('confidence_score', ascending=False)

        # Save to CSV
        try:
            csv_path = outputFolder + 'eccm_convergence_confidence_summary.csv'
            df_confidence2.to_csv(csv_path, index=False)
            print(f"\n✓ Saved ECCM convergence confidence scores to: {csv_path}")
        except Exception as e:
            print(f"\n!  Warning: Could not save ECCM confidence CSV: {e}")

        # Calculate statistics
        n_total = len(df_confidence2)
        n_high = (df_confidence2['confidence_category'] == 'high').sum()
        n_medium = (df_confidence2['confidence_category'] == 'medium').sum()
        n_low = (df_confidence2['confidence_category'] == 'low').sum()
        n_converged_binary = df_confidence2['convergence_binary'].sum()

        print(f"\nTotal ECCM pairs analyzed: {n_total}")
        print(f"Converged (binary check): {n_converged_binary}")
        print(f"\nConfidence Distribution:")
        print(f"  High confidence (>0.7):    {n_high:3d} ({n_high/n_total*100:5.1f}%)")
        print(f"  Medium confidence (0.4-0.7): {n_medium:3d} ({n_medium/n_total*100:5.1f}%)")
        print(f"  Low confidence (<0.4):     {n_low:3d} ({n_low/n_total*100:5.1f}%)")

        # Show top 5 highest confidence pairs
        if len(df_confidence2) > 0:
            print("\nTop 5 highest confidence ECCM convergence pairs:")
            print("-" * 70)
            top5 = df_confidence2.head(5)
            for idx, row in top5.iterrows():
                cat_marker = "✓✓" if row['confidence_category'] == 'high' else "✓" if row['confidence_category'] == 'medium' else "○"
                print(f"  {cat_marker} {row['x1']:20s} → {row['x2']:20s} | "
                      f"score={row['confidence_score']:.3f}, late_skill={row['late_mean']:.3f}")

        print("="*70)
    else:
        print("\n!  Warning: No ECCM convergence confidence data to save")


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
    df = df.fillna(0)
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
    parser.add_argument('--conv_skill_threshold', type=float, default=0.1, help='Minimum skill level required for convergence')
    parser.add_argument('--conv_variance_threshold', type=float, default=0.1, help='Maximum variance allowed for convergence')
    parser.add_argument('--conv_min_improvement', type=float, default=0.02, help='Minimum improvement from early to recent for convergence')
    parser.add_argument('--prefer_zero_lag', type=str, default='true', help='Prefer immediate effects (lag 0) when meaningful, or always use strongest effect')


    args = parser.parse_args()

    print("File Path:", args.file_path)
    print("Output Folder:", args.output_folder)
    print("Target Column:", args.target_column)

    df = load_data(args.file_path)
    process_data(df, args.output_folder, args.target_column, args.confounders.split(','), args.subSetLength, args.jumpN, args.z_score_threshold, args.resample_freq,
                 args.embedding_dim, args.lag, args.eccm_window_size, args.number_of_cores, args.file_path, args.ccm_training_proportion, args.max_mi_shift, args.check_convergence,
                 args.conv_skill_threshold, args.conv_variance_threshold, args.conv_min_improvement)
    
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
