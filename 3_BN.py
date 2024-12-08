#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ofir
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import bnlearn as bn
import networkx as nx
from itertools import groupby
from operator import itemgetter
import pydot
from sklearn.metrics import confusion_matrix, accuracy_score
#from random import randrange
from sklearn.metrics import roc_auc_score
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import mannwhitneyu
import pickle


def to_str(l):
    return str("_".join(l))

def discreteBounds(l):
    ranges = []
    for k, g in groupby(enumerate(l), lambda x: x[0]-x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        ranges.append((group[0], group[-1]))
        return ranges

def setBounds(p, d):
    l = []
    for i in p:
        x = d[i][-1]
        l.append(x)
    return l

parser = argparse.ArgumentParser(
    description="Process data for CCM and ECCM analysis.")
parser.add_argument('--file_path', type=str, required=True,
                    help='Path to the input CSV file')
parser.add_argument('--output_folder', type=str, required=True,
                    help='Path to tunique output folder each run')
parser.add_argument('--target_column', type=str,
                    required=True, help='Target column for analysis')
parser.add_argument('--confounders', type=str, default="",
                    help='Comma-separated list of confounder columns')
parser.add_argument('--subSetLength', type=int, default=60,
                    help='Subset length for analysis')
parser.add_argument('--jumpN', type=int, default=30,
                    help='Jump N value for processing')
parser.add_argument('--z_score_threshold', type=float,
                    default=3.0, help='Z-score threshold for outlier detection')
parser.add_argument('--resample_freq', type=str, default='1M',
                    help='Frequency for data resampling')
parser.add_argument('--embedding_dim', type=int, default=2,
                    help='Embedding dimension for CCM')
parser.add_argument('--lag', type=int, default=1, help='Lag for CCM')
parser.add_argument('--number_of_cores', type=int, default=1,
                    help='Number of cores for multithreading')
parser.add_argument('--ccm_training_proportion', type=float,
                    default=0.75, help='CCM training proportion in CCM calculation')
parser.add_argument('--max_mi_shift', type=int, default=20,
                    help='Max mutual information shift')
parser.add_argument('--auto_categorization', type=str, default='',
                    help='Mode for auto-categorization (auto or "")')
parser.add_argument('--categorization', type=str, default='',
                    help='Categorization file path if categorization mode is upload')
#parser.add_argument('--restrain_edges', type=str, default='', help='Restrain edges file path if restrain edges mode is upload')
parser.add_argument('--bn_training_fraction', type=float,
                    default=0.75, help='Training fraction for BN')
parser.add_argument('--number_of_random_vecs', type=int,
                    default=100, help='Number of random vectors')
parser.add_argument('--probability_cutoff', type=float,
                    default=0.5, help='Probability cutoff for edge inclusion')
parser.add_argument('--bidirectional_interaction', type=str, default='higher',
                    help='For DAG keep higher or earlier effect edge when bidirectional')


args = parser.parse_args()

file_path = str(args.file_path)
print(f"file_path: {file_path}" , flush=True)

target_column = str(args.target_column)
print(f"target_column: {target_column}", flush=True)

confounders = str(args.confounders)
print(f"confounders: {confounders}", flush=True)

subSetLength = int(args.subSetLength)
print(f"subSetLength: {subSetLength}", flush=True)

jumpN = int(args.jumpN)
print(f"jumpN: {jumpN}", flush=True)

z_score_threshold = float(args.z_score_threshold)
print(f"z_score_threshold: {z_score_threshold}", flush=True)

resample_freq = str(args.resample_freq)
print(f"resample_freq: {resample_freq}", flush=True)

embedding_dim = int(args.embedding_dim)
print(f"embedding_dim: {embedding_dim}", flush=True)

lag = int(args.lag)
print(f"lag: {lag}", flush=True)

number_of_cores = int(args.number_of_cores)
print(f"number_of_cores: {number_of_cores}", flush=True)

ccm_training_proportion = float(args.ccm_training_proportion)
print(f"ccm_training_proportion: {ccm_training_proportion}", flush=True)

max_mi_shift = int(args.max_mi_shift)
print(f"max_mi_shift: {max_mi_shift}", flush=True)

auto_categorization = str(args.auto_categorization)  # auto or ""
print(f"auto_categorization: {auto_categorization}", flush=True)

categorization = str(args.categorization)
print(f"categorization: {categorization}", flush=True)

bn_training_fraction = float(args.bn_training_fraction)
print(f"bn_training_fraction: {bn_training_fraction}", flush=True)

number_of_random_vecs = int(args.number_of_random_vecs)
print(f"number_of_random_vecs: {number_of_random_vecs}", flush=True)

probability_cutoff = float(args.probability_cutoff)
print(f"probability_cutoff: {probability_cutoff}", flush=True)

output_folder = str(args.output_folder)+"/"
print(f"output_folder: {output_folder}", flush=True)

bidirectional_interaction = str(args.bidirectional_interaction)
print(f"bidirectional_interaction: {bidirectional_interaction}", flush=True)

# global font size
plt.rcParams['font.size'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24


BaseFolder = "./"

targetlist = [args.target_column]

confounders = args.confounders.split(',')

concated_ = pd.read_csv(file_path)

try:
    concated_['Date'] = pd.to_datetime(concated_['Date'])
    concated_ = concated_.set_index('Date')
    date_col = 'Date'
except:
    concated_['date'] = pd.to_datetime(concated_['date'])
    concated_ = concated_.set_index('date')
    date_col = 'date'

Full_cols = list(concated_.columns)

# Interpolated not normalized for categorization
df_interpolatedNotNormalized = concated_.copy()

# Normalize 0-1
df_upsampled_normalized = pd.DataFrame(index=concated_.index)
#df_upsampled_normalized = df_concated_smoothed.copy()
AllScalersDict = {}
for i in concated_.columns:
    scaler = MinMaxScaler((0, 1))
    scaled_data = scaler.fit_transform(concated_[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_concated_fixed_outlayers = df_upsampled_normalized.copy()

# fix outlayers
for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 3)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(
        mask).interpolate()
    df_interpolatedNotNormalized[i] = df_interpolatedNotNormalized[i].mask(
        mask).interpolate()  # (method='polynomial', order=2)

try:
    df_CausalFeatures2 = pd.read_csv(output_folder+"/Surr_filtered.csv")
except:
    df_CausalFeatures2 = pd.read_csv(output_folder+"/CCM_ECCM_curated.csv")
    
df_CausalFeatures2_untouched = df_CausalFeatures2.copy()

df_CausalFeatures2 = df_CausalFeatures2[~df_CausalFeatures2['x2'].isin(confounders)]
df_CausalFeatures2 = df_CausalFeatures2[~df_CausalFeatures2['x1'].isin(targetlist)]

for i in df_CausalFeatures2.columns:
    if "Unnamed" in i:
        try:
            del df_CausalFeatures2[i]

        except:
            print()

try:
    del df_CausalFeatures2['x1x2']
except:
    print()


try:
    del df_CausalFeatures2['Score_quantile']
except:
    print()

Cols = list(df_CausalFeatures2['x1'].unique()) + \
    list(df_CausalFeatures2['x2'].unique())
Cols = list(set(Cols))
Cols = [i.replace('_', ' ') for i in Cols]

# make the network as DAG, not cyclic
df_CausalFeatures2_dag = df_CausalFeatures2.copy()

df_CausalFeatures2_dag = df_CausalFeatures2_dag[~df_CausalFeatures2_dag['x2'].isin(confounders)]
df_CausalFeatures2_dag = df_CausalFeatures2_dag[~df_CausalFeatures2_dag['x1'].isin(targetlist)]

df_CausalFeatures2_dag = df_CausalFeatures2_dag[df_CausalFeatures2_dag["is_Valid"] == 2]

def filter_bidirectional_interactions(df, criterion='higher'):
    # Identify bidirectional interactions
    bidirectional_pairs = set()
    keep_indices = set()

    for i, row in df.iterrows():
        pair = (row['x1'], row['x2'])
        reverse_pair = (row['x2'], row['x1'])

        if reverse_pair in bidirectional_pairs:
            # Found a bidirectional pair
            reverse_index = df[(df['x1'] == row['x2']) &
                               (df['x2'] == row['x1'])].index[0]

            if criterion == 'higher':
                if row['Score'] > df.at[reverse_index, 'Score']:
                    keep_indices.add(i)
                else:
                    keep_indices.add(reverse_index)
            elif criterion == 'earlier':
                if row['timeToEffect'] < df.at[reverse_index, 'timeToEffect']:
                    keep_indices.add(i)
                else:
                    keep_indices.add(reverse_index)
        else:
            bidirectional_pairs.add(pair)
            keep_indices.add(i)

    filtered_df = df.loc[keep_indices]
    return filtered_df


df_CausalFeatures2_dag = filter_bidirectional_interactions(df_CausalFeatures2_dag, bidirectional_interaction)

G_dag = nx.from_pandas_edgelist(df_CausalFeatures2_dag, 'x1', 'x2', create_using=nx.DiGraph())

# create DAG from network.
G_dag_tmp = G_dag.copy()
trimmed = []
s = 0
while s == 0:
    try:
        cycles = nx.find_cycle(G_dag_tmp)

        if (cycles[0][0] in confounders):
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
        elif (cycles[-1][0] in targetlist) and not (cycles[-1][1] in targetlist):
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
        elif (cycles[-1][1] in targetlist) and not (cycles[0][1] in targetlist):
            G_dag_tmp.remove_edge(cycles[0][0], cycles[0][1])
        elif (cycles[0][1] in targetlist) and not (cycles[-1][1] in targetlist):
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
        elif (cycles[0][0] in targetlist):
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
        else:
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
    except:
        print()
        s = 1

# Trim edge nodes (no out-edges)
s = 0
while s == 0:
    try:
        remove = [node for node, degree in dict(G_dag_tmp.out_degree()).items() if (
            degree == 0) and not (node in targetlist)]
        #remove += [node for node,degree in dict(G_dag_tmp.in_degree()).items() if (degree == 0) and not (node in targetlist)]
        G_dag_tmp.remove_nodes_from(remove)
        if len(remove) == 0:
            s = 1
    except:
        print()
        s = 1


edges = G_dag_tmp.edges
DAG = bn.make_DAG(list(edges))

df = df_interpolatedNotNormalized.dropna()[list(G_dag_tmp.nodes)].dropna().copy()
df = df.resample(resample_freq).interpolate("linear")

# fix outlayers
for i in df.columns:
    mask = (np.abs(stats.zscore(df[i])) > 3)
    df[i] = df[i].mask(mask).interpolate(method='polynomial', order=2)

#df[df < 0] = 0
df = df.dropna()

AllScalersDict = {}
for i in df.columns:
    scaler = MinMaxScaler((0, 1))
    scaled_data = scaler.fit_transform(df[i].values.reshape(-1, 1))
    df[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_cut = pd.DataFrame()
cols = list(df.columns)
cols_remove = []

if auto_categorization == "auto":

    # here scan for best vectors for each of the affected vars, keep in vecs dict that will be used for categorization
    def generate_quantile_vectors(min_gap=0.05, n_categories=4):
        quantile_vectors = []
        step = min_gap
        if n_categories == 4:
            for q1 in np.arange(step, 1 - (n_categories - 2) * step, step):
                for q2 in np.arange(q1 + step, 1 - (n_categories - 3) * step, step):
                    for q3 in np.arange(q2 + step, 1 - (n_categories - 4) * step, step):
                        quantile_vectors.append([0, q1, q2, q3, 1])
        if n_categories == 3:
            for q1 in np.arange(step, 1 - (n_categories - 2) * step, step):
                for q2 in np.arange(q1 + step, 1 - (n_categories - 3) * step, step):
                    quantile_vectors.append([0, q1, q2, 1])
        if n_categories == 2:
            for q1 in np.arange(step, 1 - (n_categories - 2) * step, step):
                quantile_vectors.append([0, q1, 1])

        return quantile_vectors

    dict_vecs = {}

    quantile_vectors_var = generate_quantile_vectors(min_gap=0.05, n_categories=3)
    quantile_vectors_var_fixed = []

    quantile_vectors_target = generate_quantile_vectors(min_gap=0.05, n_categories=2)
    quantile_vectors_target_fixed = []

    for vec in quantile_vectors_var:
        vec = [float(Decimal(str(n)).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP)) for n in vec]
        quantile_vectors_var_fixed.append(vec)

    for vec in quantile_vectors_target:
        vec = [float(Decimal(str(n)).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP)) for n in vec]
        quantile_vectors_target_fixed.append(vec)

    for variable in df.columns:
        if variable not in targetlist:
            
            # FIND OPTIMAL CATEGORIZATION INTO 3 categories
            results = []
            for vec in quantile_vectors_var_fixed:
                
                significant_pairs_count = 0
                if len(list(set(vec))) == len(vec):

                    quantile_groups_var = df.groupby(pd.cut(df[variable], bins=vec, include_lowest=False))
                    groups_var = list(quantile_groups_var.groups.keys())

                    for i in range(len(groups_var)):
                        for j in range(i + 1, len(groups_var)):
                            try:
                                group1 = quantile_groups_var.get_group(groups_var[i])[variable]
                                group2 = quantile_groups_var.get_group(groups_var[j])[variable]
                                statistic, p_value = mannwhitneyu(group1.values, group2.values)
                                
                                common_values = np.intersect1d(group1.values, group2.values)
                                num_common_values = len(common_values)
                                if num_common_values > 0:
                                    p_value = 1
                                
                            except:
                                p_value = 1
                            
                            # Check if the p-value is significant (e.g., p < 0.05)
                            if (p_value <= 0.05) : 

                                significant_pairs_count += 1
                    
                    results.append([vec, significant_pairs_count])

            max_significant_pairs = max(results, key=lambda x: x[1])[1]
            best_vectors = [result for result in results if result[1] == max_significant_pairs]

            def calculate_evenness(vec):
                return np.std(np.diff(vec))

            best_var_vector = min(best_vectors, key=lambda x: calculate_evenness(x[0]))
            dict_vecs[variable] = best_var_vector

            # Plotting the results
            indices = list(range(len(results)))
            significant_counts = [result[1] for result in results]

            plt.figure(figsize=(5, 5))
            plt.plot(indices, significant_counts, marker='o', linestyle='')
            plt.xlabel('Vector Index')
            plt.ylabel('Total Number of Significant Results')
            plt.title('Total Number of Significant Results for '+variable)
            plt.grid(True)
            plt.savefig(output_folder + variable + "_vectorScan.png")
            plt.close()

            print(
                f"Best Quantile Vector: {best_var_vector[0]} with {best_var_vector[1]} significant pairs")

            quantile_filename = output_folder + variable + "_quantiles.csv"

            # Retrieve original data for quantile computation
            df_cut[variable] = pd.cut(df[variable], bins=best_var_vector[0], labels=['0', '1', '2'], include_lowest=True)

            # Compute quantiles
            quantiles = df[variable].quantile(q=best_var_vector[0])
            quantiles.to_frame().to_csv(quantile_filename)

        else:
            # FIND OPTIMAL CATEGORIZATION INTO 2 categories for the TARGET
            results = []
            for vec in quantile_vectors_target_fixed:
                significant_pairs_count = 0
                if len(list(set(vec))) == len(vec):

                    quantile_groups_var = df.groupby(
                        pd.cut(df[variable], bins=vec, include_lowest=True))
                    groups_var = list(quantile_groups_var.groups.keys())

                    for i in range(len(groups_var)):
                        for j in range(i + 1, len(groups_var)):
                            try:
                                group1 = quantile_groups_var.get_group(groups_var[i])[variable]
                                group2 = quantile_groups_var.get_group(groups_var[j])[variable]
                                statistic, p_value = mannwhitneyu(group1.values, group2.values)
                            except:
                                p_value = 1

                            # Check if the p-value is significant (e.g., p < 0.05)
                            if (p_value <= 0.05):

                                significant_pairs_count += 1
                    
                    results.append([vec, significant_pairs_count])

            max_significant_pairs = max(results, key=lambda x: x[1])[1]
            best_vectors = [result for result in results if result[1] == max_significant_pairs]

            def calculate_evenness(vec):
                return np.std(np.diff(vec))

            best_target_vector = min(
                best_vectors, key=lambda x: calculate_evenness(x[0]))
            dict_vecs[variable] = best_target_vector

            quantiles = df[variable].quantile(q=best_target_vector[0])

            quantile_filename = output_folder + variable + "_quantiles.csv"

            quantiles.to_frame().to_csv(quantile_filename)

            df_cut[variable] = pd.cut(df[variable].values, bins=best_target_vector[0], labels=['0', '1'], include_lowest=True)

else:
    df_categories = pd.read_csv(categorization)
    for variable in df.columns:
        df_tmp = df_categories[df_categories["variable"] == variable]
        print(df_tmp)
        vec = df_tmp.values.tolist()[0][1].split(";")
        vec = [float(i) for i in vec]
        if len(vec) == 3:
            labels = ['0', '1']
        elif len(vec) == 4:
            labels = ['0', '1', '2']
        df_cut[variable] = pd.cut(
            df[variable].values, bins=vec, labels=labels, include_lowest=True)


# - shift columns according to - timeToEffect
#timetoeffect - dict
df_CausalFeatures2 = df_CausalFeatures2_dag.copy()

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] > 0]
df_CausalFeatures2 = df_CausalFeatures2.dropna()
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["is_Valid"] == 2]


G_dag = nx.DiGraph()

for i in df_CausalFeatures2.values.tolist():
    G_dag.add_edge(i[0], i[1], weight=i[4])

lengths = dict(nx.all_pairs_dijkstra_path_length(G_dag))

dict_allLags = {}

for j in targetlist:
    for i in cols:
        try:
            dict_allLags[(i, j)] = lengths[i][j]
        except:
            print("missing interaction")

# loop over columns and shift according to timetoeffect dict
dict_df_cuts = {}
for i in targetlist:
    df_tmp = df_cut.copy()
    for j in cols:
        try:
            s = dict_allLags[(j, i)]
            # fix shift according to interpolation. if weekly, use s as is
            df_tmp[j] = df_tmp[j].shift(int(s))
        except:
            print("missing interaction")
    df_tmp = df_tmp.dropna()
    dict_df_cuts[i] = df_tmp


dict_acc = {}

for t in targetlist:
    # Split test - train
    df_cut = dict_df_cuts[t].sample(frac=0.75, random_state=42)
    df_cut_test = dict_df_cuts[t].drop(df_cut.index)

    # make testset balanced
    column = t
    df_cut_test = df_cut_test.groupby(column).sample(n=df_cut_test[column].value_counts().min(), random_state=42)

    edges = list(G_dag_tmp.edges)
    
    DAG = bn.make_DAG(list(edges))

    nodes = list(DAG['adjmat'].columns)

    DAG_global = bn.parameter_learning.fit(DAG, df_cut[nodes], methodtype='bayes')
    dict_df_cuts[t+"_dag_global"] = DAG_global
    # For comparison - learn structure from data using the causal features
    DAG_global_learned = bn.structure_learning.fit(df_cut[nodes])
    dict_df_cuts[t+"_dag_global_learned"] = DAG_global_learned

    # validate
    dict_test = {}
    l = [list(i) for i in DAG_global['model_edges']]
    model_nodes = [item for sublist in l for item in sublist]
    model_nodes = list(set(model_nodes))

    cases = df_cut_test[model_nodes].values.tolist()
    keys = model_nodes

    all_p = []
    for i, vali in enumerate(cases):
        dict_test = {}
        for j, valj in enumerate(keys):
            dict_test[valj] = str(vali[j])

        for j in targetlist:
            try:
                del dict_test[j]
            except:
                print()
        try:
            q1 = bn.inference.fit(DAG_global, variables=[t], evidence=dict_test)
            all_p.append(q1.df.p[1])
        except:    
            all_p.append(0.5)

    df_test = pd.DataFrame()
    df_test['Observed'] = df_cut_test[t].values.tolist()
    df_test['Predicted'] = all_p
    df_test = df_test.astype(float)

    plt.figure(figsize=(6, 6))
    ax = df_test.reset_index().plot(kind="scatter", s=30, x="index", y="Predicted", c="orange", figsize=(6, 6))
    df_test.reset_index().plot(kind="scatter", x="index", y="Observed", secondary_y=False, ax=ax)
    plt.ylabel('Probability', fontsize=24)
    plt.xlabel('Test samples', fontsize=24)
    #plt.title("BN Model Validation", fontsize=24)
    plt.savefig(output_folder + 'BN_model_validation.png', bbox_inches='tight', transparent=True)
    plt.close()

     
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x=df_test["Observed"], y=df_test["Predicted"], ax=ax, boxprops={"facecolor": (.4, .6, .8, .5)})
    ax.set_xlabel("Observed", fontsize=24)
    ax.set_ylabel("Predicted", fontsize=24)
    plt.tight_layout()
    plt.savefig(output_folder + 'BN_model_results.png', bbox_inches='tight', transparent=True)
    plt.close()

    def roundProb(p):
        if p >= probability_cutoff:
            return 1
        else:
            return 0

    # create confusion matrix
    
    df_test['p_binary'] = df_test['Predicted'].apply(roundProb)
    
    acc = accuracy_score(df_test['Observed'].values, df_test['p_binary'].values)
    cm = confusion_matrix(df_test['Observed'].values, df_test['p_binary'].values)
        
    # Transpose the confusion matrix to swap the axes
    cm_transposed = cm.T
    
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=2)  
    sns.heatmap(cm_transposed, annot=True, cmap="Blues", fmt="d")
    plt.xlabel('Observed', fontsize=24)
    plt.ylabel('Predicted', fontsize=24)
    plt.savefig(output_folder + 'BN_model_confusionMatrix.png', bbox_inches='tight', transparent=True)
    plt.close()

    print(t+" acc = " + str(acc))
    dict_acc[t] = acc

    AllNodes = [item for sublist in DAG_global['model_edges']
                for item in sublist]
    AllNodes = list(set(AllNodes))
    AllEdges = edges

    g = pydot.Dot()

    for node in AllNodes:
        # if node in environmentalCols:
        #    g.add_node(pydot.Node( node, color='orange', style='filled'))
        if node in targetlist:
            g.add_node(pydot.Node(node, color='cyan', style='filled'))
        else:
            g.add_node(pydot.Node(node, color='orange', style='filled'))

    for i in AllEdges:
        g.add_edge(pydot.Edge(i[0],
                              i[1],
                              color='black',
                              # style='dashed'
                              ))
    g.set_size('"6,6!"')
    g.write_png(output_folder+"CausalDAG_NET.png")

    try:
        rocauc = roc_auc_score(df_test['p_binary'].values, df_test['Observed'].values)
        print("roc_auc  " + str(rocauc))
    except:
        rocauc = -1



def f_max(v):
    #make integer
    v = [round(i) for i in v]
    print(str(v))
    dict_evidence = {}
    l = [list(i) for i in DAG_global['model_edges']]
    model_nodes = [item for sublist in l for item in sublist]    
    for j, valj in enumerate(path[:-1]):
        try:
            if (valj in model_nodes) and (valj in AllNodes):
                    dict_evidence[valj] = str(v[j])
        except:
            print('*')
            
    for j in targetlist:
        try:
            del dict_evidence[j]
        except:
            print('**')
    print(dict_evidence) 
    q1 = bn.inference.fit(DAG_global, variables=[path[-1]], evidence=dict_evidence)
    df_q = q1.df
    #minimize min probability
    df_q_reduced = df_q[df_q[path[-1]] == 0]
    if len(df_q_reduced) == 0:
        df_q_reduced = df_q[df_q[path[-1]] == '0']
    if len(df_q_reduced) > 0:
        score = df_q_reduced.p[0]   
    else:
       score = 0
   
    return 1-score 

res_sub_max = []
#res_sub_min = []

for t in targetlist:
    DAG_global = dict_df_cuts[t+'_dag_global']

    AllNodes = [item for sublist in DAG_global['model_edges']
                for item in sublist]
    AllNodes = list(set(AllNodes))

    dictBounds = {}
    dict_NodesUniqueValues = {}

    for j in dict_df_cuts[t].columns:
        unq = list(dict_df_cuts[t][j].unique())
        unq = [int(k) for k in unq]

        dictBounds[j] = discreteBounds(unq)
        dict_NodesUniqueValues[j] = [str(u) for u in unq]

    #res_sub = []
    path = AllNodes
    path = [i for i in path if i != t]
    path = path+[t]
    ###

    bounds = setBounds(path, dictBounds)

    # create  permutations.
# =============================================================================
#     vec = []
#     listOfRandomVecs = []
#     
#     number_of_random_vecs = 0#10*len(df_cut)
#     
#     for j in range(0, number_of_random_vecs):
#         for k, valk in enumerate(bounds):
#             r = 4
#             while (r >= bounds[k][0]) and (r <= bounds[k][1]):
#                 r = randrange(3)
#             vec.append(randrange(3))
#         listOfRandomVecs.append(vec)
#         vec = []
# =============================================================================

# here replace listOfRandomVecs with df_cut, which is the whole dataset. 
    
    listOfRandomVecs = df_cut[path].astype(int).values.tolist()
    # max
    for j in listOfRandomVecs:
        try:
            result = f_max(j[:-1])
            v = [round(i) for i in j[:-1]]
            print(str(v))
            dict_evidence = {}
            l = [list(i) for i in DAG_global['model_edges']]
            model_nodes = [item for sublist in l for item in sublist]    
            for j, valj in enumerate(path[:-1]):
                try:
                    if (valj in model_nodes) and (valj in AllNodes):
                            dict_evidence[valj] = str(v[j])
                except:
                    print()
                    
            for j in targetlist:
                try:
                    del dict_evidence[j]
                except:
                    print()
            print(dict_evidence) 
            res_sub_max.append([path[-1], dict_evidence, result])
        except:
            print()


# Read all PM results. arrange as DF
max_listofscores = res_sub_max.copy()

# prep for Df
allmax = []

for i in max_listofscores:
    tmp = list(i[1].values())
    tmp.append(i[2])
    tmp.append(i[0])
    allmax.append(tmp)

df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max_vecs = df_de_max[df_de_max['Score'] > probability_cutoff]
df_de_min_vecs = df_de_max[df_de_max['Score'] < probability_cutoff]


# FIGURE
cols = df_de_max_vecs.columns
cols = [i for i in cols if not i in ['Score', 'y']]
df_de_max_vecs = df_de_max_vecs[cols].astype(float)


# FIGURE
df_CausalFeatures2 = df_CausalFeatures2.dropna()
df_de_max = df_de_max_vecs.copy()
plt.close()

for t in targetlist:
    
    l = df_de_max[cols].astype(int).mean().reset_index().values.tolist()
    #l = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean().reset_index().values.tolist()
    
    df_mean = df_de_max[cols].astype(int).mean()
    #df_mean = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean()
    
    #ll, _ = fix_labels([i[0] for i in l])
    #df_mean.index = ll
    l = df_mean.reset_index().values.tolist()
    # inverse for visualization
    d = {}
    for i in l:
        try:
            d[i[0]] = 1/i[1]
        except:
            d[i[0]] = 0

    scaler = MinMaxScaler(feature_range=(0, 0.458))

    scaler.fit(np.array(list(d.values())).reshape(-1, 1))
    X = scaler.transform(np.array(list(d.values())).reshape(-1, 1))
    for k, valk in enumerate(d.keys()):
        d[valk] = X[k][0]

    g = pydot.Dot()
    AllNodes_ = AllNodes
    #AllNodes_, _ = fix_labels(AllNodes_)

    edgesL_ = [i[0] for i in edges]
    edgesR_ = [i[1] for i in edges]
    #edgesL_, _ = fix_labels(edgesL_)
    #edgesR_, _ = fix_labels(edgesR_)
    edges_ = [(edgesL_[i], edgesR_[i]) for i in range(0, len(edgesL_))]

    for node in cols:
        if not node in t:
            nd = pydot.Node(node,
                            style='filled',
                            fontsize="20pt",
                            fillcolor=str(d[node])+" 1 1")
            g.add_node(nd)

    for c, i in enumerate(edges):
        
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) & (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'black'
        else:
            is_direct = 'gray'

        if lbl == 0:
            lbl = '<'+resample_freq
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(lbl)

        g.add_edge(pydot.Edge(edges_[c][0],
                              edges_[c][1],
                              color=is_direct,
                              style="filled",
                              label=lbl,
                              fontsize="20pt"
                              ))
    g.set_size('"6,6!"')
    g.write_png(output_folder+"CausalDAG_NET_MAX.png")


# Mean min

df_de_min = df_de_min_vecs.copy()

for t in targetlist:
    # l = df_de_min[[i for i in AllNodes if not i in targetlist]].astype(int).mean().reset_index().values.tolist()
    l = df_de_min[cols].astype(int).mean().reset_index().values.tolist()
    #df_mean = df_de_min[[i for i in AllNodes if not i in targetlist]].astype(int).mean()
    df_mean = df_de_min[cols].astype(int).mean()
    
    #ll, _ = fix_labels([i[0] for i in l])
    #df_mean.index = ll
    l = df_mean.reset_index().values.tolist()
    # inverse for visualization
    d = {}
    for i in l:
        try:
            d[i[0]] = 1/i[1]
        except:
            d[i[0]] = 0

    scaler = MinMaxScaler(feature_range=(0, 0.458))

    scaler.fit(np.array(list(d.values())).reshape(-1, 1))
    X = scaler.transform(np.array(list(d.values())).reshape(-1, 1))
    for k, valk in enumerate(d.keys()):
        d[valk] = X[k][0]

    g = pydot.Dot()
    AllNodes_ = AllNodes
    #AllNodes_, _ = fix_labels(AllNodes_)

    edgesL_ = [i[0] for i in edges]
    edgesR_ = [i[1] for i in edges]
    #edgesL_, _ = fix_labels(edgesL_)
    #edgesR_, _ = fix_labels(edgesR_)
    edges_ = [(edgesL_[i], edgesR_[i]) for i in range(0, len(edgesL_))]

    for node in cols:
        if not node in t:
            nd = pydot.Node(node,
                            style='filled',
                            fontsize="20pt",
                            fillcolor=str(d[node])+" 1 1")
            g.add_node(nd)

    for c, i in enumerate(edges):
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) & (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'black'
        else:
            is_direct = 'gray'

        if lbl == 0:
            lbl = '<'+resample_freq
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(lbl)

        g.add_edge(pydot.Edge(edges_[c][0],
                              edges_[c][1],
                              color=is_direct,
                              style="filled",
                              label=lbl,
                              fontsize="20pt"
                              ))
    g.set_size('"6,6!"')
    g.write_png(output_folder+"CausalDAG_NET_MIN.png")

df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
#df_de_max = df_de_max[df_de_max['Score'] != probability_cutoff]


#######Figures#########
for t in targetlist:
    DAG_global_learned = dict_df_cuts[t+"_dag_global_learned"]
    learned_dags_djmat = DAG_global_learned['adjmat']*1
   
    plt.figure(figsize=(6, 6))
    g = sns.clustermap(learned_dags_djmat, cbar=False, col_cluster=False, row_cluster=False, linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    #plt.title("Interactions learned from CCM features")
       
    # Set axis labels using the correct axes
    g.ax_heatmap.set_xlabel("Source", fontsize=24, labelpad=20)
    g.ax_heatmap.set_ylabel("Target", fontsize=24, labelpad=20)
        
    # Remove the colorbar explicitly (if Seaborn creates it by default)
    if g.cax is not None:
        g.cax.set_visible(False)    
    plt.savefig(output_folder + 'learned_fromCCMfeatures_dag.png', bbox_inches='tight', transparent=True)
    plt.close()

    #######

    # compare networks - CCM and learned
    dict_df_cuts[t+"_dag_global"]
    ccm_dags_djmat = DAG_global['adjmat']*1

    plt.figure(figsize=(6, 6))
   
    g = sns.clustermap(ccm_dags_djmat, cbar=False, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,
                       )  # row_colors=row_colors, col_colors=col_colors)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    # Set axis labels using the correct axes
    g.ax_heatmap.set_xlabel("Source", fontsize=24, labelpad=20)
    g.ax_heatmap.set_ylabel("Target", fontsize=24, labelpad=20)
        
    # Remove the colorbar explicitly (if Seaborn creates it by default)
    if g.cax is not None:
        g.cax.set_visible(False)    
    plt.title("DAG based on the interactions identified by CCM")
    plt.savefig(output_folder+'ccm_dag.png', bbox_inches='tight', transparent=True)
    plt.close()

    #######

try:
    df_CausalFeatures2 = pd.read_csv(output_folder+"/Surr_filtered.csv")
except:
    df_CausalFeatures2 = pd.read_csv(output_folder+"/CCM_ECCM_curated.csv")
    
ccm_eccm = df_CausalFeatures2.pivot(index='x1', columns='x2', values='Score')


# General settings for all plots
plt.rcParams['figure.figsize'] = (6, 6)  
plt.rcParams['xtick.labelsize'] = 24  
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['font.size'] = 24  
sns.set(style="white") 

# Visualization: Clustermap for CCM ECCM
g = sns.clustermap(ccm_eccm.fillna(0), cbar=True, col_cluster=False, row_cluster=False,
                   linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)


# Remove dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)

# Ensure colorbar is aligned and of the same height
g.cax.set_position([.97, .2, .03, .45])  

plt.savefig(output_folder+'ccm_eccm.png', bbox_inches='tight', transparent=True)
plt.close()

# Sensitivity analysis
df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()

def calculate_diff(lst, mean_output):
    return [abs(x - mean_output) for x in lst]


def mean_contribution(inputs, output):
    inputs = np.asarray(inputs)
    new_inputs = []
    for c, i in enumerate(inputs):
        new_inputs.append([int(j) for j in i])

    # Calculate the mean output
    mean_output = np.mean(output)
    num_samples, num_vars = inputs.shape
    df_tmp = pd.DataFrame(data=new_inputs)
    df_tmp["y"] = output
    df_vars_contributions = pd.DataFrame()
    df_vars_std = pd.DataFrame()  

    # Iterate over each input variable
    for i in range(num_vars):

        df_lists = df_tmp.groupby(i)['y'].aggregate(list).reset_index()
        # Apply the function to calculate the difference to the mean output
        df_lists['diff_to_mean_output'] = df_lists['y'].apply(
            lambda x: [y - mean_output for y in x])
        # Calculate mean and std of contributions
        df_vars_contributions[i] = df_lists['diff_to_mean_output'].apply(
            lambda x: sum(x) / len(x))
        df_vars_std[i] = df_lists['diff_to_mean_output'].apply(
            lambda x: np.std(x))

    return df_vars_contributions, df_vars_std


inputs = df_de_max[[i for i in df_de_max.columns if not i in [
    "Score", "y"]]].values.tolist()
output = df_de_max["Score"].values.tolist()

mean_contributions, std_contributions = mean_contribution(inputs, output)
mean_contributions = mean_contributions.fillna(0)
std_contributions = std_contributions.fillna(0)

xticks = [i for i in df_de_max.columns if not i in ["Score", "y"]]

# Sum the mean and std contributions
sum_mean_contributions = mean_contributions.abs().sum(axis=0)
sum_std_contributions = std_contributions.abs().sum(axis=0)
sum_mean_contributions.index = xticks
sum_std_contributions.index = xticks

# Create bar plot with standard deviation bars
sum_mean_contributions.index = xticks
sum_std_contributions.index = xticks

# Bar plot with error bars
plt.figure(figsize=(6, 6))
sum_mean_contributions.sort_values().plot(kind="bar", capsize=4, color='blue')

plt.xlabel('Input Variable', fontsize=16)
plt.ylabel('Sum of |Mean Contribution|', fontsize=16)
#plt.title("Sensitivity", fontsize=18)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.savefig(output_folder + "sensitivity_barplot.png", bbox_inches='tight', dpi=600)
plt.close()

bn.save(DAG_global, filepath=output_folder+'bnlearn_model', overwrite=True)

#Save essentials for later inference
dict_model_essentials = {}
dict_model_essentials["nodes"] = path[:-1]
dict_model_essentials["target"] = path[-1]
dict_model_essentials["accuracy"] = acc
dict_model_essentials["roc_auc"] = rocauc

with open(output_folder+'dict_model_essentials.pickle', 'wb') as handle:
    pickle.dump(dict_model_essentials, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(output_folder+'bounds.pickle', 'wb') as handle:
    pickle.dump(dictBounds, handle, protocol=pickle.HIGHEST_PROTOCOL)














