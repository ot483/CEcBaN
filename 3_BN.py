#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ofir
"""

import argparse
import numpy as np
import pandas as pd
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
import datetime  
import json      
import os      

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns



def save_scenarios_and_frequencies_for_users_safe(output_dir, high_scenario_list, low_scenario_list, 
                                                 high_freq_list, low_freq_list, target_var, column_list):
    """
    Save scenario and frequency data in user-friendly formats
    SAFE VERSION - Uses different parameter names to avoid variable shadowing
    """
    
    local_output_dir = str(output_dir)
    local_high_scenarios = list(high_scenario_list) if high_scenario_list else []
    local_low_scenarios = list(low_scenario_list) if low_scenario_list else []
    local_high_frequencies = list(high_freq_list) if high_freq_list else []
    local_low_frequencies = list(low_freq_list) if low_freq_list else []
    local_target_column = str(target_var)
    local_cols = list(column_list) if column_list else []
    
    
    all_scenarios_data = []
    
    for i, (scenario, frequency) in enumerate(zip(local_high_scenarios, local_high_frequencies)):
        scenario_row = {
            'scenario_id': f'HIGH_{i+1:03d}',
            'scenario_type': 'High Probability',
            'target_variable': local_target_column,
            'frequency_percent': round(frequency, 2),
            'scenario_rank': i + 1
        }
        for var_name in local_cols:
            if var_name in scenario:
                scenario_row[f'{var_name}_value'] = scenario[var_name]
                scenario_row[f'{var_name}_category'] = ['Low', 'Medium', 'High'][scenario[var_name]] if scenario[var_name] in [0,1,2] else 'Unknown'
        
        all_scenarios_data.append(scenario_row)
    
    for i, (scenario, frequency) in enumerate(zip(local_low_scenarios, local_low_frequencies)):
        scenario_row = {
            'scenario_id': f'LOW_{i+1:03d}',
            'scenario_type': 'Low Probability', 
            'target_variable': local_target_column,
            'frequency_percent': round(frequency, 2),
            'scenario_rank': i + 1
        }
        for var_name in local_cols:
            if var_name in scenario:
                scenario_row[f'{var_name}_value'] = scenario[var_name]
                scenario_row[f'{var_name}_category'] = ['Low', 'Medium', 'High'][scenario[var_name]] if scenario[var_name] in [0,1,2] else 'Unknown'
        
        all_scenarios_data.append(scenario_row)
    
    scenarios_csv_path = None
    if all_scenarios_data:
        df_all_scenarios = pd.DataFrame(all_scenarios_data)
        scenarios_csv_path = os.path.join(local_output_dir, 'all_scenarios_with_frequencies.csv')
        df_all_scenarios.to_csv(scenarios_csv_path, index=False)
        print(f"✓ Saved comprehensive scenarios to: {scenarios_csv_path}")
    
    summary_data = []
    
    if local_high_scenarios:
        high_freq_stats = {
            'scenario_type': 'High Probability',
            'total_scenarios': len(local_high_scenarios),
            'min_frequency': round(min(local_high_frequencies), 2) if local_high_frequencies else 0,
            'max_frequency': round(max(local_high_frequencies), 2) if local_high_frequencies else 0,
            'mean_frequency': round(sum(local_high_frequencies) / len(local_high_frequencies), 2) if local_high_frequencies else 0,
            'total_frequency_coverage': round(sum(local_high_frequencies), 2)
        }
        summary_data.append(high_freq_stats)
    
    if local_low_scenarios:
        low_freq_stats = {
            'scenario_type': 'Low Probability',
            'total_scenarios': len(local_low_scenarios),
            'min_frequency': round(min(local_low_frequencies), 2) if local_low_frequencies else 0,
            'max_frequency': round(max(local_low_frequencies), 2) if local_low_frequencies else 0,
            'mean_frequency': round(sum(local_low_frequencies) / len(local_low_frequencies), 2) if local_low_frequencies else 0,
            'total_frequency_coverage': round(sum(local_low_frequencies), 2)
        }
        summary_data.append(low_freq_stats)
    
    summary_csv_path = None
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(local_output_dir, 'scenario_frequency_summary.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Saved scenario summary to: {summary_csv_path}")
    
    report_lines = []
    report_lines.append(f"SCENARIO AND FREQUENCY ANALYSIS REPORT")
    report_lines.append(f"=" * 50)
    report_lines.append(f"Target Variable: {local_target_column}")
    report_lines.append(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    if local_high_scenarios:
        report_lines.append(f"HIGH PROBABILITY SCENARIOS ({len(local_high_scenarios)} total)")
        report_lines.append("-" * 30)
        high_sorted = sorted(zip(local_high_scenarios, local_high_frequencies, range(len(local_high_scenarios))), 
                           key=lambda x: x[1], reverse=True)
        
        for i, (scenario, freq, orig_idx) in enumerate(high_sorted[:10]):  # Top 10
            report_lines.append(f"Rank {i+1}: Frequency {freq:.1f}%")
            var_descriptions = []
            for var_name, val in scenario.items():
                cat = ['Low', 'Medium', 'High'][val] if val in [0,1,2] else f'Value_{val}'
                var_descriptions.append(f"{var_name}: {cat}")
            report_lines.append(f"  Variables: {' | '.join(var_descriptions)}")
            report_lines.append("")
        
        if len(local_high_scenarios) > 10:
            report_lines.append(f"... and {len(local_high_scenarios) - 10} more high probability scenarios")
            report_lines.append("")
    
    if local_low_scenarios:
        report_lines.append(f"LOW PROBABILITY SCENARIOS ({len(local_low_scenarios)} total)")
        report_lines.append("-" * 30)
        low_sorted = sorted(zip(local_low_scenarios, local_low_frequencies, range(len(local_low_scenarios))), 
                          key=lambda x: x[1], reverse=True)
        
        for i, (scenario, freq, orig_idx) in enumerate(low_sorted[:10]):  # Top 10
            report_lines.append(f"Rank {i+1}: Frequency {freq:.1f}%")
            var_descriptions = []
            for var_name, val in scenario.items():
                cat = ['Low', 'Medium', 'High'][val] if val in [0,1,2] else f'Value_{val}'
                var_descriptions.append(f"{var_name}: {cat}")
            report_lines.append(f"  Variables: {' | '.join(var_descriptions)}")
            report_lines.append("")
        
        if len(local_low_scenarios) > 10:
            report_lines.append(f"... and {len(local_low_scenarios) - 10} more low probability scenarios")
            report_lines.append("")
    
    report_path = os.path.join(local_output_dir, 'scenario_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f" Saved readable report to: {report_path}")
    
    json_data = {
        'analysis_info': {
            'target_variable': local_target_column,
            'analysis_date': datetime.datetime.now().isoformat(),
            'total_high_scenarios': len(local_high_scenarios),
            'total_low_scenarios': len(local_low_scenarios)
        },
        'high_probability_scenarios': [
            {
                'scenario_id': f'HIGH_{i+1:03d}',
                'frequency_percent': freq,
                'variables': scenario
            } for i, (scenario, freq) in enumerate(zip(local_high_scenarios, local_high_frequencies))
        ],
        'low_probability_scenarios': [
            {
                'scenario_id': f'LOW_{i+1:03d}',
                'frequency_percent': freq,
                'variables': scenario  
            } for i, (scenario, freq) in enumerate(zip(local_low_scenarios, local_low_frequencies))
        ]
    }
    
    json_path = os.path.join(local_output_dir, 'scenarios_and_frequencies.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved JSON data to: {json_path}")
    
    return {
        'csv_path': scenarios_csv_path,
        'summary_path': summary_csv_path,
        'report_path': report_path,
        'json_path': json_path
    }




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

def filter_bidirectional_interactions(df, criterion='higher'):

    bidirectional_pairs = set()
    keep_indices = set()

    for i, row in df.iterrows():
        pair = (row['x1'], row['x2'])
        reverse_pair = (row['x2'], row['x1'])

        if reverse_pair in bidirectional_pairs:

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

def create_principled_dag(df_edges, target_columns, confounders, output_folder="./"):
    """
    Create DAG following exact logic rules:
    1. Discard sink nodes (except targets)
    2. Find feedback loops
    3. Cut feedback loop far from target, close to confounders
    4. Use CCM score to decide where to cut
    5. Iterate until no sink nodes or feedback loops
    """
    print(f"Creating DAG from {len(df_edges)} edges...")
    

    G = nx.from_pandas_edgelist(df_edges, 'x1', 'x2', 
                                edge_attr=['Score', 'timeToEffect'], 
                                create_using=nx.DiGraph())
    
    removed_edges = []
    removed_nodes = []
    iteration = 0
    max_iterations = len(df_edges) * 2  
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current: {len(G.nodes())} nodes, {len(G.edges())} edges")
        

        sink_nodes = [node for node in G.nodes() 
                     if G.out_degree(node) == 0 and node not in target_columns]
        
        if sink_nodes:
            print(f"Removing {len(sink_nodes)} sink nodes: {sink_nodes}")
            G.remove_nodes_from(sink_nodes)
            removed_nodes.extend(sink_nodes)
            continue  
        

        if nx.is_directed_acyclic_graph(G):
            print("Graph is now a DAG!")
            break
        

        try:

            cycle = nx.find_cycle(G, orientation='original')
            cycle_nodes = [u for u, v, d in cycle]
            cycle_edges = [(u, v) for u, v, d in cycle]
            
            print(f"Found cycle with {len(cycle_nodes)} nodes: {cycle_nodes}")
            

            best_edge_to_cut = find_best_cycle_edge_to_cut(
                G, cycle_edges, target_columns, confounders, df_edges
            )
            
            if best_edge_to_cut:
                u, v = best_edge_to_cut
                edge_data = G[u][v]
                score = edge_data.get('Score', 0)
                time_effect = edge_data.get('timeToEffect', 0)
                
                G.remove_edge(u, v)
                removed_edges.append({
                    'from': u, 'to': v, 'score': score, 
                    'time_effect': time_effect, 'iteration': iteration
                })
                
                print(f"Cut edge: {u} -> {v} (score: {score:.3f})")
            else:
                print("Could not find edge to cut")
                break
                
        except nx.NetworkXNoCycle:
            print("No cycles found")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    

    isolated = [node for node in G.nodes() if G.degree(node) == 0]
    if isolated:
        G.remove_nodes_from(isolated)
        removed_nodes.extend(isolated)
    
    print(f"\n=== Final Result ===")
    print(f"Iterations: {iteration}")
    print(f"Removed {len(removed_edges)} edges, {len(removed_nodes)} nodes")
    print(f"Final: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
    

    final_sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    target_sinks = [n for n in final_sinks if n in target_columns]
    other_sinks = [n for n in final_sinks if n not in target_columns]
    
    print(f"Target sinks: {target_sinks}")
    if other_sinks:
        print(f"WARNING: Other sinks: {other_sinks}")
    

    if removed_edges:
        removed_df = pd.DataFrame(removed_edges)
        try:
            removed_df.to_csv(output_folder + "removed_edges_analysis.csv", index=False)
        except:
            pass
    
    return G, removed_edges

def find_best_cycle_edge_to_cut(G, cycle_edges, target_columns, confounders, df_edges):
    """
    Find best edge to cut in cycle following the CORRECTED logic:
    1. Never cut edges TO targets
    2. Cut far from target (maximize distance to target)
    3. Cut close to confounders (minimize distance from confounders)
    4. Use CCM score as tiebreaker (cut lower scores)
    """
    if not cycle_edges:
        return None
    

    safe_edges = [(u, v) for u, v in cycle_edges if v not in target_columns]
    if not safe_edges:
        print("WARNING: All cycle edges go to targets!")
        safe_edges = cycle_edges  
    
    print(f"Safe edges to consider: {safe_edges}")
    
    best_edge = None
    best_distance_to_target = -1
    best_distance_from_confounder = float('inf') 
    best_score = 1.0  
    
    for u, v in safe_edges:
        edge_data = G[u][v]
        score = edge_data.get('Score', 0)
        

        dist_from_confounder = get_min_distance_from_confounders(G, u, confounders)  # Changed to min distance
        dist_to_target = get_min_distance_to_targets(G, v, target_columns)
        
        print(f"Edge {u}->{v}: score={score:.3f}, dist_from_conf={dist_from_confounder}, dist_to_targ={dist_to_target}")
        

        should_select = False
        reason = ""
        

        if dist_to_target > best_distance_to_target:
            should_select = True
            reason = f"farther from target ({dist_to_target})"
        elif dist_to_target == best_distance_to_target and dist_to_target > 0:

            if confounders and dist_from_confounder < best_distance_from_confounder:
                should_select = True
                reason = f"same distance to target, closer to confounder ({dist_from_confounder})"
            elif confounders and dist_from_confounder == best_distance_from_confounder:

                if score < best_score:
                    should_select = True
                    reason = f"same distances, lower score ({score:.3f})"
            elif not confounders:

                if score < best_score:
                    should_select = True
                    reason = f"same distance to target, lower score ({score:.3f})"
        
        if should_select:
            best_edge = (u, v)
            best_distance_to_target = dist_to_target
            best_distance_from_confounder = dist_from_confounder
            best_score = score
            print(f"  -> NEW BEST: {reason}")
    
    if best_edge:
        print(f"Selected edge to cut: {best_edge[0]} -> {best_edge[1]}")
    
    return best_edge

def get_min_distance_from_confounders(G, node, confounders):
    """Get minimum distance from any confounder to the node (closer to confounders is better)"""
    if not confounders:
        return float('inf') 
    
    min_dist = float('inf')
    for conf in confounders:
        if conf in G.nodes():
            try:
                dist = nx.shortest_path_length(G, conf, node)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
    return min_dist if min_dist != float('inf') else float('inf')

def get_min_distance_to_targets(G, node, target_columns):
    """Get minimum distance from node to any target"""
    if not target_columns:
        return 0
    
    min_dist = float('inf')
    for target in target_columns:
        if target in G.nodes():
            try:
                dist = nx.shortest_path_length(G, node, target)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
    
    return min_dist if min_dist != float('inf') else 0

def generate_distinct_colors(values, variable_names, scenario_type="max"):
    """Generate distinct HSV colors for better visualization"""

    if len(values) == 0:
        return {}
    
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        normalized_values = [0.5] * len(values)
    else:
        normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
    
    colors = {}
    
    if scenario_type == "max":

        for i, norm_val in enumerate(normalized_values):
            if i < len(variable_names):
                key = variable_names[i]

                hue = 0.67 * (1 - norm_val) 
                saturation = 1.0  
                value = 1.0  
                colors[key] = f"{hue:.3f} {saturation:.3f} {value:.3f}"
    else:  

        for i, norm_val in enumerate(normalized_values):
            if i < len(variable_names):
                key = variable_names[i]
                if norm_val <= 0.5:

                    hue = 0.33 - (0.16 * (norm_val / 0.5))
                else:

                    hue = 0.17 - (0.17 * ((norm_val - 0.5) / 0.5))
                
                saturation = 1.0  
                value = 1.0 
                colors[key] = f"{hue:.3f} {saturation:.3f} {value:.3f}"
    
    return colors

def calculate_scenario_frequencies(scenarios, df_frequency_reference, cols):
    """Calculate how often each scenario appears in the reference dataset"""
    frequencies = []
    
    if len(scenarios) == 0 or df_frequency_reference is None or len(df_frequency_reference) == 0:
        return frequencies
    
    total_rows = len(df_frequency_reference)
    

    debug_file = output_folder + "frequency_calculation_debug.txt"
    with open(debug_file, 'w') as f:
        f.write("=== CALCULATE_SCENARIO_FREQUENCIES DEBUG ===\n")
        f.write(f"Total rows in reference: {total_rows}\n")
        f.write(f"Number of scenarios: {len(scenarios)}\n\n")
        
        for idx, scenario in enumerate(scenarios):
            f.write(f"--- Scenario {idx+1}: {scenario} ---\n")
            
            matching_rows = 0
            for row_idx, row in df_frequency_reference.iterrows():
                matches_all = True
                for col, val in scenario.items():
                    if col in df_frequency_reference.columns:
                        try:
                            if int(row[col]) != int(val):
                                matches_all = False
                                break
                        except (ValueError, TypeError):
                            matches_all = False
                            break
                    else:
                        matches_all = False
                        break
                if matches_all:
                    matching_rows += 1
            
            frequency = (matching_rows / total_rows) * 100 if total_rows > 0 else 0
            frequencies.append(frequency)
            
            f.write(f"Matching rows: {matching_rows}\n")
            f.write(f"Frequency: {frequency:.1f}%\n\n")
    
    return frequencies

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


df_interpolatedNotNormalized = concated_.copy()


df_upsampled_normalized = pd.DataFrame(index=concated_.index)

AllScalersDict = {}
for i in concated_.columns:
    scaler = MinMaxScaler((0, 1))
    scaled_data = scaler.fit_transform(concated_[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_concated_fixed_outlayers = df_upsampled_normalized.copy()


for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 3)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(
        mask).interpolate()
    df_interpolatedNotNormalized[i] = df_interpolatedNotNormalized[i].mask(
        mask).interpolate()  

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


df_CausalFeatures2_dag = df_CausalFeatures2.copy()

df_CausalFeatures2_dag = df_CausalFeatures2_dag[~df_CausalFeatures2_dag['x2'].isin(confounders)]
df_CausalFeatures2_dag = df_CausalFeatures2_dag[~df_CausalFeatures2_dag['x1'].isin(targetlist)]

df_CausalFeatures2_dag = df_CausalFeatures2_dag[df_CausalFeatures2_dag["is_Valid"] == 2]

df_CausalFeatures2_dag = filter_bidirectional_interactions(df_CausalFeatures2_dag, bidirectional_interaction)

G_dag = nx.from_pandas_edgelist(df_CausalFeatures2_dag, 'x1', 'x2', create_using=nx.DiGraph())

G_dag_tmp, removed_edges_info = create_principled_dag(
    df_CausalFeatures2_dag, targetlist, confounders, output_folder
)

edges = G_dag_tmp.edges
DAG = bn.make_DAG(list(edges))

df = df_interpolatedNotNormalized.dropna()[list(G_dag_tmp.nodes)].dropna().copy()
df = df.resample(resample_freq).interpolate("linear")


for i in df.columns:
    mask = (np.abs(stats.zscore(df[i])) > 3)
    df[i] = df[i].mask(mask).interpolate(method='polynomial', order=2)


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
                            

                            if (p_value <= 0.05) : 

                                significant_pairs_count += 1  
                    
                    results.append([vec, significant_pairs_count])

            max_significant_pairs = max(results, key=lambda x: x[1])[1]
            best_vectors = [result for result in results if result[1] == max_significant_pairs]

            def calculate_evenness(vec):
                return np.std(np.diff(vec))

            best_var_vector = min(best_vectors, key=lambda x: calculate_evenness(x[0]))
            dict_vecs[variable] = best_var_vector


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


            df_cut[variable] = pd.cut(df[variable], bins=best_var_vector[0], labels=['0', '1', '2'], include_lowest=True)


            quantiles = df[variable].quantile(q=best_var_vector[0])
            quantiles.to_frame().to_csv(quantile_filename)

        else:

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
    df_categories = pd.read_csv(categorization, skipinitialspace=True)
    
    for variable in df.columns:
        df_tmp = df_categories[df_categories["variable"] == variable]
    
        if df_tmp.empty:
            print(f"Warning: No categorization found for variable '{variable}'")
            continue
        
        try:
            vec = df_tmp.iloc[0, 1].split(";") 
            vec = [float(i) for i in vec] 
        except Exception as e:
            print(f"Error processing bins for '{variable}': {e}")
            continue
    
        labels = [str(i) for i in range(len(vec) - 1)]
    
        df_cut[variable] = pd.cut(df[variable], bins=vec, labels=labels, include_lowest=True)



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
            df_tmp[j] = df_tmp[j].shift(int(s))
        except:
            print("missing interaction")
    df_tmp = df_tmp.dropna()
    dict_df_cuts[i] = df_tmp


dict_acc = {}

for t in targetlist:
   
    dag_nodes = list(DAG['adjmat'].columns)
 
    
    available_cols = set(dict_df_cuts[t].columns)
    required_cols = set(dag_nodes)
    missing_cols = required_cols - available_cols
    extra_cols = available_cols - required_cols
    

    
    available_dag_cols = [col for col in dag_nodes if col in dict_df_cuts[t].columns]
    if len(available_dag_cols) != len(dag_nodes):
        print(f"WARNING: Only {len(available_dag_cols)}/{len(dag_nodes)} DAG columns available in data")
    

    df_cut = dict_df_cuts[t][available_dag_cols].sample(frac=float(bn_training_fraction), random_state=42)
    df_cut_test = dict_df_cuts[t][available_dag_cols].drop(df_cut.index)


    column = t
    df_cut_test = df_cut_test.groupby(column).sample(n=df_cut_test[column].value_counts().min(), random_state=42)

    edges = list(G_dag_tmp.edges)
    
    DAG = bn.make_DAG(list(edges))

    nodes = list(DAG['adjmat'].columns)
    

    DAG_global = bn.parameter_learning.fit(DAG, df_cut[nodes], methodtype='bayes')
    dict_df_cuts[t+"_dag_global"] = DAG_global

    DAG_global_learned = bn.structure_learning.fit(df_cut[nodes])
    dict_df_cuts[t+"_dag_global_learned"] = DAG_global_learned


    dict_test = {}
    l = [list(i) for i in DAG_global['model_edges']]
    model_nodes = [item for sublist in l for item in sublist]
    model_nodes = list(set(model_nodes))
   
    

    available_test_nodes = [node for node in model_nodes if node in df_cut_test.columns]

    cases = df_cut_test[available_test_nodes].values.tolist()
    keys = available_test_nodes

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

    
    df_test['p_binary'] = df_test['Predicted'].apply(roundProb)
    
    acc = accuracy_score(df_test['Observed'].values, df_test['p_binary'].values)
    cm = confusion_matrix(df_test['Observed'].values, df_test['p_binary'].values)
        
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

    AllNodes = list(DAG_global['adjmat'].columns)
    
    AllEdges = edges

    g = pydot.Dot()

    for node in AllNodes:
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
    g.set_size('"12,12!"')
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


    trained_model_nodes = list(DAG_global['adjmat'].columns)
    
    # Construct path using trained model node names
    path = [node for node in trained_model_nodes if node != t]
    path = path + [t]

    bounds = setBounds(path, dictBounds)

    
    missing_cols = set(path) - set(df_cut.columns)
    if missing_cols:
        print(f"ERROR: df_cut missing path columns: {missing_cols}")

        available_path_cols = [col for col in path if col in df_cut.columns]
        if available_path_cols:
            listOfRandomVecs = df_cut[available_path_cols].astype(int).values.tolist()
        else:
            print(f"ERROR: No path columns available in df_cut")
            listOfRandomVecs = []
    else:
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


max_listofscores = res_sub_max.copy()

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
df_de_max_vecs_filtered = df_de_max_vecs[cols].astype(float)

df_CausalFeatures2 = df_CausalFeatures2.dropna()
df_de_max = df_de_max_vecs.copy()  # Keep original with Score column
df_de_min = df_de_min_vecs.copy()
plt.close()

high_scenarios = []
low_scenarios = []

if len(df_de_max_vecs) > 0:
    for _, row in df_de_max_vecs.iterrows():
        scenario = {col: int(row[col]) for col in cols}
        high_scenarios.append(scenario)

if len(df_de_min_vecs) > 0:
    for _, row in df_de_min_vecs.iterrows():
        scenario = {col: int(row[col]) for col in cols}
        low_scenarios.append(scenario)

shifted_reference_data = dict_df_cuts[t].copy()  


# Write debug info to file instead of print
debug_file = output_folder + "frequency_debug.txt"
with open(debug_file, 'w') as f:
    f.write("=== FREQUENCY DEBUG ===\n")
    f.write(f"Reference data shape: {shifted_reference_data.shape}\n")
    f.write(f"Reference data columns: {list(shifted_reference_data.columns)}\n")
    
    if high_scenarios:
        f.write(f"Number of high scenarios: {len(high_scenarios)}\n")
        f.write(f"First scenario: {high_scenarios[0]}\n")
        
        # Save first few rows of reference data
        f.write("First 5 rows of reference data:\n")
        f.write(shifted_reference_data.head(5).to_string())
        f.write("\n\n")
        
        # Manual check for first scenario
        first_scenario = high_scenarios[0]
        manual_matches = 0
        for idx, row in shifted_reference_data.iterrows():
            matches = True
            for col, val in first_scenario.items():
                if col in shifted_reference_data.columns:
                    if int(row[col]) != int(val):
                        matches = False
                        break
                else:
                    matches = False
                    break
            if matches:
                manual_matches += 1
        
        f.write(f"Manual frequency check: {manual_matches} out of {len(shifted_reference_data)} = {(manual_matches/len(shifted_reference_data)*100):.1f}%\n")

print(f"Debug info written to: {debug_file}")

high_frequencies = calculate_scenario_frequencies(high_scenarios, shifted_reference_data, cols) if high_scenarios else []
low_frequencies = calculate_scenario_frequencies(low_scenarios, shifted_reference_data, cols) if low_scenarios else []


for t in targetlist:
    
    if len(df_de_max) > 0:

        df_mean_max = df_de_max_vecs[cols].astype(int).mean()
        l_max = df_mean_max.reset_index().values.tolist()
        print(f"DEBUG: MEAN MAX scenario values: {df_mean_max.to_dict()}")
    else:
        l_max = [(col, 1) for col in cols]  
    

    values_list_max = [val for _, val in l_max]
    variable_names_max = [var for var, _ in l_max]
    d_max = generate_distinct_colors(values_list_max, variable_names_max, "max")

    
    g_max = pydot.Dot()
    
    edgesL_ = [i[0] for i in edges]
    edgesR_ = [i[1] for i in edges]
    edges_ = [(edgesL_[i], edgesR_[i]) for i in range(0, len(edgesL_))]

    for node in cols:
        if node not in t:
            nd = pydot.Node(node,
                            style='filled',
                            fontsize="20pt",
                            fillcolor=d_max[node])
            g_max.add_node(nd)

    for c, i in enumerate(edges):
        matching_rows = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) & (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()
        if len(matching_rows) > 0:
            lbl = matching_rows[0]
        else:
            lbl = 0  
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

        g_max.add_edge(pydot.Edge(edges_[c][0],
                              edges_[c][1],
                              color=is_direct,
                              style="filled",
                              label=lbl,
                              fontsize="20pt"))
    
    g_max.set_size('"12,12!"')
    g_max.write_png(output_folder+"CausalDAG_NET_MEAN_MAX.png")

    if len(df_de_min) > 0:
        # Calculate MEAN of all low probability scenarios
        df_mean_min = df_de_min_vecs[cols].astype(int).mean()
        l_min = df_mean_min.reset_index().values.tolist()
    else:
        l_min = [(col, 1) for col in cols] 
        
    values_list_min = [val for _, val in l_min]
    variable_names_min = [var for var, _ in l_min]
    d_min = generate_distinct_colors(values_list_min, variable_names_min, "min")

    g_min = pydot.Dot()

    for node in cols:
        if node not in t:
            nd = pydot.Node(node,
                            style='filled',
                            fontsize="20pt",
                            fillcolor=d_min[node])
            g_min.add_node(nd)

    for c, i in enumerate(edges):
        matching_rows = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) & (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()
        if len(matching_rows) > 0:
            lbl = matching_rows[0]
        else:
            lbl = 0         
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

        g_min.add_edge(pydot.Edge(edges_[c][0],
                              edges_[c][1],
                              color=is_direct,
                              style="filled",
                              label=lbl,
                              fontsize="20pt"))
    
    g_min.set_size('"12,12!"')
    g_min.write_png(output_folder+"CausalDAG_NET_MEAN_MIN.png")

    
    for i, scenario in enumerate(high_scenarios):
        l_individual = [(col, scenario[col]) for col in cols]
        values_list = [val for _, val in l_individual]
        variable_names = [var for var, _ in l_individual]
        d_individual = generate_distinct_colors(values_list, variable_names, "max")
        
        g_individual = pydot.Dot()
        
        for node in cols:
            if node not in t:
                nd = pydot.Node(node,
                                style='filled',
                                fontsize="20pt",
                                fillcolor=d_individual[node])
                g_individual.add_node(nd)

        for c, edge in enumerate(edges):
            try:
                lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == edge[0]) & (df_CausalFeatures2['x2'] == edge[1])]['timeToEffect'].values.tolist()[0]
            except (IndexError, KeyError):
                lbl = 0  # Default value when edge not found            
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

            g_individual.add_edge(pydot.Edge(edges_[c][0],
                                  edges_[c][1],
                                  color=is_direct,
                                  style="filled",
                                  label=lbl,
                                  fontsize="20pt"))
        
        g_individual.set_size('"12,12!"')
        g_individual.write_png(output_folder+f"scenario_high_{i:03d}.png")


    for i, scenario in enumerate(low_scenarios):
        l_individual = [(col, scenario[col]) for col in cols]
        values_list = [val for _, val in l_individual]
        variable_names = [var for var, _ in l_individual]
        d_individual = generate_distinct_colors(values_list, variable_names, "min")
        
        g_individual = pydot.Dot()
        
        for node in cols:
            if node not in t:
                nd = pydot.Node(node,
                                style='filled',
                                fontsize="20pt",
                                fillcolor=d_individual[node])
                g_individual.add_node(nd)

        for c, edge in enumerate(edges):
            try:
                lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == edge[0]) & (df_CausalFeatures2['x2'] == edge[1])]['timeToEffect'].values.tolist()[0]
            except (IndexError, KeyError):
                lbl = 0  # Default value when edge not found
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

            g_individual.add_edge(pydot.Edge(edges_[c][0],
                                  edges_[c][1],
                                  color=is_direct,
                                  style="filled",
                                  label=lbl,
                                  fontsize="20pt"))
        
        g_individual.set_size('"12,12!"')
        g_individual.write_png(output_folder+f"scenario_low_{i:03d}.png")

df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()



#######Figures#########
for t in targetlist:
    DAG_global_learned = dict_df_cuts[t+"_dag_global_learned"]
    learned_dags_djmat = DAG_global_learned['adjmat']*1
   
    plt.figure(figsize=(6, 6))
    g = sns.clustermap(learned_dags_djmat, cbar=False, col_cluster=False, row_cluster=False, linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

       

    g.ax_heatmap.set_xlabel("Target", fontsize=24, labelpad=20)
    g.ax_heatmap.set_ylabel("Source", fontsize=24, labelpad=20)
        
    
    if g.cax is not None:
        g.cax.set_visible(False)    
    plt.savefig(output_folder + 'learned_fromCCMfeatures_dag.png', bbox_inches='tight', transparent=True)
    plt.close()

    #######


    dict_df_cuts[t+"_dag_global"]
    ccm_dags_djmat = DAG_global['adjmat']*1

    plt.figure(figsize=(6, 6))
   
    g = sns.clustermap(ccm_dags_djmat, cbar=False, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,
                       ) 
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xlabel("Target", fontsize=24, labelpad=20)
    g.ax_heatmap.set_ylabel("Source", fontsize=24, labelpad=20)
        

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



plt.rcParams['figure.figsize'] = (6, 6)  
plt.rcParams['xtick.labelsize'] = 24  
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['font.size'] = 24  
sns.set(style="white") 


g = sns.clustermap(ccm_eccm.fillna(0), cbar=True, col_cluster=False, row_cluster=False,
                   linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)




g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)


g.cax.set_position([.97, .2, .03, .45])  

plt.savefig(output_folder+'ccm_eccm.png', bbox_inches='tight', transparent=True)
plt.close()


df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()

def calculate_diff(lst, mean_output):
    return [abs(x - mean_output) for x in lst]


def mean_contribution(inputs, output):
    inputs = np.asarray(inputs)
    new_inputs = []
    for c, i in enumerate(inputs):
        new_inputs.append([int(j) for j in i])


    mean_output = np.mean(output)
    num_samples, num_vars = inputs.shape
    df_tmp = pd.DataFrame(data=new_inputs)
    df_tmp["y"] = output
    df_vars_contributions = pd.DataFrame()
    df_vars_std = pd.DataFrame()  


    for i in range(num_vars):

        df_lists = df_tmp.groupby(i)['y'].aggregate(list).reset_index()

        df_lists['diff_to_mean_output'] = df_lists['y'].apply(
            lambda x: [y - mean_output for y in x])

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


sum_mean_contributions = mean_contributions.abs().sum(axis=0)
sum_std_contributions = std_contributions.abs().sum(axis=0)
sum_mean_contributions.index = xticks
sum_std_contributions.index = xticks


sum_mean_contributions.index = xticks
sum_std_contributions.index = xticks


plt.figure(figsize=(6, 6))
sum_mean_contributions.sort_values().plot(kind="bar", capsize=4, color='blue')

plt.xlabel('Input Variable', fontsize=16)
plt.ylabel('Sum of |Mean Contribution|', fontsize=16)

plt.grid(False)
plt.gca().set_facecolor('white')
plt.savefig(output_folder + "sensitivity_barplot.png", bbox_inches='tight', dpi=600)
plt.close()

bn.save(DAG_global, filepath=output_folder+'bnlearn_model', overwrite=True)


dict_model_essentials = {}
dict_model_essentials["nodes"] = path[:-1]
dict_model_essentials["target"] = path[-1]
dict_model_essentials["accuracy"] = acc
dict_model_essentials["roc_auc"] = rocauc

try:
    test_evidence = {node: '1' for node in dict_model_essentials['nodes'][:2]}
    print(f"Testing inference with saved node names: {test_evidence}")
    test_q = bn.inference.fit(DAG_global, variables=[dict_model_essentials['target']], evidence=test_evidence)
    print(f"Inference test SUCCESS with saved names")
except Exception as e:
    print(f"Inference test FAILED with saved names: {e}")

with open(output_folder+'dict_model_essentials.pickle', 'wb') as handle:
    pickle.dump(dict_model_essentials, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(output_folder+'bounds.pickle', 'wb') as handle:
    pickle.dump(dictBounds, handle, protocol=pickle.HIGHEST_PROTOCOL)

scenario_data = {}

if len(df_de_max_vecs) > 0:
    scenario_data['high_scenarios'] = high_scenarios
    scenario_data['high_frequencies'] = high_frequencies
    print(f"DEBUG: Found {len(high_scenarios)} HIGH probability scenarios")
    print(f"DEBUG: High scenario frequencies: {high_frequencies}")

if len(df_de_min_vecs) > 0:
    scenario_data['low_scenarios'] = low_scenarios
    scenario_data['low_frequencies'] = low_frequencies
   


debug_file2 = output_folder + "pickle_save_debug.txt"
with open(debug_file2, 'w') as f:
    f.write("=== PICKLE SAVE DEBUG ===\n")
    f.write(f"Number of high_scenarios: {len(high_scenarios)}\n")
    f.write(f"Number of high_frequencies: {len(high_frequencies)}\n")
    f.write(f"High frequencies: {high_frequencies}\n\n")
    
    f.write(f"Number of low_scenarios: {len(low_scenarios)}\n")
    f.write(f"Number of low_frequencies: {len(low_frequencies)}\n")
    f.write(f"Low frequencies: {low_frequencies}\n\n")
    
    f.write("First 3 high scenarios:\n")
    for i, scenario in enumerate(high_scenarios[:3]):
        freq = high_frequencies[i] if i < len(high_frequencies) else "NO_FREQ"
        f.write(f"  Scenario {i+1}: {scenario} -> {freq}%\n")


with open(output_folder+'scenario_data.pickle', 'wb') as handle:
    pickle.dump(scenario_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

try:
    if 'high_scenarios' in locals() and 'low_scenarios' in locals() and 'cols' in locals():
        print("Attempting to save user-friendly files...")
        safe_file_paths = save_scenarios_and_frequencies_for_users_safe(
            output_folder, high_scenarios, low_scenarios, 
            high_frequencies, low_frequencies, target_column, cols
        )
        print(" User-friendly files saved successfully")
    else:
        print(" Required variables not available for user files")
except Exception as e:
    print(f" Could not save user files (model still works): {e}")
    import traceback
    traceback.print_exc()