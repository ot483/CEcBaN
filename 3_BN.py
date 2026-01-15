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
from sklearn.model_selection import StratifiedKFold
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
    FIX 2.2: Enhanced DAG construction with comprehensive logging

    Create DAG following exact logic rules:
    1. Discard sink nodes (except targets)
    2. Find feedback loops
    3. Cut feedback loop far from target, close to confounders
    4. Use CCM score to decide where to cut
    5. Iterate until no sink nodes or feedback loops

    All decisions logged to CSV and summary text file for transparency.
    """
    print("\n" + "="*70)
    print("DAG CONSTRUCTION WITH TRANSPARENCY LOGGING")
    print("="*70)
    print(f"Starting with {len(df_edges)} edges...")
    print(f"Target variables: {target_columns}")
    print(f"Confounders: {confounders}")


    G = nx.from_pandas_edgelist(df_edges, 'x1', 'x2',
                                edge_attr=['Score', 'timeToEffect'],
                                create_using=nx.DiGraph())

    removed_edges = []
    removed_nodes = []
    iteration = 0
    max_iterations = len(df_edges) * 2

    # FIX 2.2: Comprehensive logging structure
    dag_log = []  # Will be saved to CSV  
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # Log iteration state
        dag_log.append({
            'iteration': iteration,
            'action': 'iteration_start',
            'n_nodes': len(G.nodes()),
            'n_edges': len(G.edges()),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'details': f"Starting iteration {iteration}"
        })


        sink_nodes = [node for node in G.nodes()
                     if G.out_degree(node) == 0 and node not in target_columns]

        if sink_nodes:
            print(f"Removing {len(sink_nodes)} sink nodes: {sink_nodes}")

            # Log each sink node removal
            for node in sink_nodes:
                in_degree = G.in_degree(node)
                dag_log.append({
                    'iteration': iteration,
                    'action': 'remove_sink_node',
                    'node': node,
                    'in_degree': in_degree,
                    'out_degree': 0,
                    'reason': f"Sink node (no outgoing edges) not in targets",
                    'details': f"Removed {node}: {in_degree} incoming edges, 0 outgoing"
                })

            G.remove_nodes_from(sink_nodes)
            removed_nodes.extend(sink_nodes)
            continue  
        

        if nx.is_directed_acyclic_graph(G):
            print("Graph is now a DAG!")
            dag_log.append({
                'iteration': iteration,
                'action': 'dag_achieved',
                'n_nodes': len(G.nodes()),
                'n_edges': len(G.edges()),
                'details': "Successfully created DAG"
            })
            break


        try:

            cycle = nx.find_cycle(G, orientation='original')
            cycle_nodes = [u for u, v, d in cycle]
            cycle_edges = [(u, v) for u, v, d in cycle]

            print(f"Found cycle with {len(cycle_nodes)} nodes: {cycle_nodes}")

            # Log cycle detection
            dag_log.append({
                'iteration': iteration,
                'action': 'cycle_detected',
                'cycle_length': len(cycle_nodes),
                'cycle_nodes': ', '.join(cycle_nodes),
                'cycle_edges': ', '.join([f"{u}->{v}" for u, v in cycle_edges]),
                'details': f"Cycle with {len(cycle_nodes)} nodes: {' -> '.join(cycle_nodes)}"
            })


            best_edge_to_cut, cut_rationale = find_best_cycle_edge_to_cut_with_logging(
                G, cycle_edges, target_columns, confounders, df_edges, iteration, dag_log
            )

            if best_edge_to_cut:
                u, v = best_edge_to_cut
                edge_data = G[u][v]
                score = edge_data.get('Score', 0)
                time_effect = edge_data.get('timeToEffect', 0)

                G.remove_edge(u, v)
                removed_edges.append({
                    'from': u, 'to': v, 'score': score,
                    'time_effect': time_effect, 'iteration': iteration,
                    'rationale': cut_rationale
                })

                print(f"Cut edge: {u} -> {v} (score: {score:.3f})")

                # Log edge removal
                dag_log.append({
                    'iteration': iteration,
                    'action': 'remove_edge',
                    'edge_from': u,
                    'edge_to': v,
                    'score': score,
                    'time_effect': time_effect,
                    'reason': cut_rationale,
                    'details': f"Removed edge {u}->{v} (score={score:.3f}): {cut_rationale}"
                })
            else:
                print("Could not find edge to cut")
                dag_log.append({
                    'iteration': iteration,
                    'action': 'error',
                    'details': "No suitable edge found to cut"
                })
                break
                
        except nx.NetworkXNoCycle:
            print("No cycles found")
            dag_log.append({
                'iteration': iteration,
                'action': 'no_cycle',
                'details': "No cycles detected"
            })
            break
        except Exception as e:
            print(f"Error: {e}")
            dag_log.append({
                'iteration': iteration,
                'action': 'error',
                'details': f"Exception: {str(e)}"
            })
            break


    isolated = [node for node in G.nodes() if G.degree(node) == 0]
    if isolated:
        for node in isolated:
            dag_log.append({
                'iteration': iteration + 1,
                'action': 'remove_isolated',
                'node': node,
                'details': f"Removed isolated node {node}"
            })
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

    # FIX 2.2: Save comprehensive logs
    print("\n" + "="*70)
    print("SAVING DAG CONSTRUCTION LOGS")
    print("="*70)

    # Save detailed log to CSV
    if dag_log:
        try:
            df_log = pd.DataFrame(dag_log)
            csv_path = output_folder + "dag_construction_log.csv"
            df_log.to_csv(csv_path, index=False)
            print(f"✓ Saved detailed log to: {csv_path}")
        except Exception as e:
            print(f"!  Warning: Could not save DAG log CSV: {e}")

    # Save human-readable summary
    try:
        summary_path = output_folder + "dag_construction_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DAG CONSTRUCTION SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Initial state:\n")
            f.write(f"  - Edges: {len(df_edges)}\n")
            f.write(f"  - Target variables: {', '.join(target_columns)}\n")
            f.write(f"  - Confounders: {', '.join(confounders) if confounders else 'None'}\n\n")

            f.write(f"Process:\n")
            f.write(f"  - Iterations: {iteration}\n")
            f.write(f"  - Nodes removed: {len(removed_nodes)} ({', '.join(removed_nodes) if removed_nodes else 'None'})\n")
            f.write(f"  - Edges removed: {len(removed_edges)}\n\n")

            f.write(f"Final state:\n")
            f.write(f"  - Nodes: {len(G.nodes())}\n")
            f.write(f"  - Edges: {len(G.edges())}\n")
            f.write(f"  - Is DAG: {nx.is_directed_acyclic_graph(G)}\n")
            f.write(f"  - Target sinks: {', '.join(target_sinks)}\n")
            if other_sinks:
                f.write(f"  - WARNING: Other sinks: {', '.join(other_sinks)}\n")
            f.write("\n")

            if removed_edges:
                f.write("="*70 + "\n")
                f.write("REMOVED EDGES DETAILS\n")
                f.write("="*70 + "\n\n")
                for idx, edge in enumerate(removed_edges, 1):
                    f.write(f"{idx}. {edge['from']} -> {edge['to']}\n")
                    f.write(f"   - Iteration: {edge['iteration']}\n")
                    f.write(f"   - Score: {edge['score']:.3f}\n")
                    f.write(f"   - Time to effect: {edge.get('time_effect', 'N/A')}\n")
                    if 'rationale' in edge:
                        f.write(f"   - Rationale: {edge['rationale']}\n")
                    f.write("\n")

            f.write("="*70 + "\n")
            f.write("ITERATION-BY-ITERATION LOG\n")
            f.write("="*70 + "\n\n")
            for log_entry in dag_log:
                f.write(f"[Iteration {log_entry.get('iteration', 'N/A')}] {log_entry.get('action', 'unknown')}\n")
                f.write(f"  {log_entry.get('details', '')}\n\n")

        print(f"✓ Saved human-readable summary to: {summary_path}")
    except Exception as e:
        print(f"!  Warning: Could not save DAG summary: {e}")

    # Save removed edges (enhanced version)
    if removed_edges:
        removed_df = pd.DataFrame(removed_edges)
        try:
            removed_path = output_folder + "removed_edges_detailed.csv"
            removed_df.to_csv(removed_path, index=False)
            print(f"✓ Saved removed edges to: {removed_path}")
        except Exception as e:
            print(f"!  Warning: Could not save removed edges CSV: {e}")

    print("="*70)

    return G, removed_edges

def find_best_cycle_edge_to_cut_with_logging(G, cycle_edges, target_columns, confounders, df_edges, iteration, dag_log):
    """
    FIX 2.2: Enhanced version with detailed logging of decision criteria

    Find best edge to cut in cycle following the CORRECTED logic:
    1. Never cut edges TO targets
    2. Cut far from target (maximize distance to target)
    3. Cut close to confounders (minimize distance from confounders)
    4. Use CCM score as tiebreaker (cut lower scores)

    Returns:
        best_edge (tuple): (u, v) edge to cut
        rationale (str): Human-readable explanation of why this edge was chosen
    """
    if not cycle_edges:
        return None, "No edges in cycle"


    safe_edges = [(u, v) for u, v in cycle_edges if v not in target_columns]
    if not safe_edges:
        print("WARNING: All cycle edges go to targets!")
        safe_edges = cycle_edges
        dag_log.append({
            'iteration': iteration,
            'action': 'warning',
            'details': "All cycle edges go to targets - will cut anyway"
        })

    print(f"Safe edges to consider: {safe_edges}")

    best_edge = None
    best_distance_to_target = -1
    best_distance_from_confounder = float('inf')
    best_score = 1.0
    best_rationale = ""

    # Log each candidate edge evaluation
    for u, v in safe_edges:
        edge_data = G[u][v]
        score = edge_data.get('Score', 0)


        dist_from_confounder = get_min_distance_from_confounders(G, u, confounders)
        dist_to_target = get_min_distance_to_targets(G, v, target_columns)

        print(f"Edge {u}->{v}: score={score:.3f}, dist_from_conf={dist_from_confounder}, dist_to_targ={dist_to_target}")

        # Log evaluation
        dag_log.append({
            'iteration': iteration,
            'action': 'evaluate_edge',
            'edge_from': u,
            'edge_to': v,
            'score': score,
            'dist_to_target': dist_to_target if dist_to_target != float('inf') else 'inf',
            'dist_from_confounder': dist_from_confounder if dist_from_confounder != float('inf') else 'inf',
            'details': f"Evaluating {u}->{v}: score={score:.3f}, dist_to_target={dist_to_target}, dist_from_conf={dist_from_confounder}"
        })


        should_select = False
        reason = ""


        if dist_to_target > best_distance_to_target:
            should_select = True
            reason = f"farther from target (dist={dist_to_target} vs {best_distance_to_target})"
        elif dist_to_target == best_distance_to_target and dist_to_target > 0:

            if confounders and dist_from_confounder < best_distance_from_confounder:
                should_select = True
                reason = f"same target distance, closer to confounder (dist={dist_from_confounder} vs {best_distance_from_confounder})"
            elif confounders and dist_from_confounder == best_distance_from_confounder:

                if score < best_score:
                    should_select = True
                    reason = f"same distances, lower CCM score ({score:.3f} vs {best_score:.3f})"
            elif not confounders:

                if score < best_score:
                    should_select = True
                    reason = f"same target distance, lower CCM score ({score:.3f} vs {best_score:.3f})"

        if should_select:
            best_edge = (u, v)
            best_distance_to_target = dist_to_target
            best_distance_from_confounder = dist_from_confounder
            best_score = score
            best_rationale = reason
            print(f"  -> NEW BEST: {reason}")

    if best_edge:
        print(f"Selected edge to cut: {best_edge[0]} -> {best_edge[1]}")
        final_rationale = (f"{best_rationale}; final metrics: "
                          f"dist_to_target={best_distance_to_target}, "
                          f"dist_from_conf={best_distance_from_confounder}, "
                          f"score={best_score:.3f}")
        return best_edge, final_rationale
    else:
        return None, "No suitable edge found"


def find_best_cycle_edge_to_cut(G, cycle_edges, target_columns, confounders, df_edges):
    """
    Original version (kept for compatibility)

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
parser.add_argument('--significance_filter_mode', type=str, default='both',
                    help='Significance filtration mode: fdr_only, surrogate_only, or both')


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
    try:
        df_CausalFeatures2 = pd.read_csv(output_folder+"/tmp/CCM_ECCM_curated.csv")
    except:
        df_CausalFeatures2 = pd.read_csv(output_folder+"/CCM_ECCM_curated.csv")
    
df_CausalFeatures2_untouched = df_CausalFeatures2.copy()

df_CausalFeatures2 = df_CausalFeatures2[~df_CausalFeatures2['x2'].isin(confounders)]
df_CausalFeatures2 = df_CausalFeatures2[~df_CausalFeatures2['x1'].isin(targetlist)]

# Apply significance filtration based on mode
significance_filter_mode = str(args.significance_filter_mode)
print(f"\nApplying significance filtration: {significance_filter_mode}", flush=True)
print(f"Edges before filtration: {len(df_CausalFeatures2)}", flush=True)

if significance_filter_mode == 'fdr_only':
    # Keep only edges with significant_fdr == True
    if 'significant_fdr' in df_CausalFeatures2.columns:
        df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['significant_fdr'] == True]
        print(f"  Using FDR only: kept {len(df_CausalFeatures2)} edges with significant_fdr=True", flush=True)
    else:
        print(f"  WARNING: significant_fdr column not found, skipping FDR filtration", flush=True)

elif significance_filter_mode == 'surrogate_only':
    # Keep only edges where Score >= Score_quantile
    if 'Score' in df_CausalFeatures2.columns and 'Score_quantile' in df_CausalFeatures2.columns:
        df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['Score'] >= df_CausalFeatures2['Score_quantile']]
        print(f"  Using surrogate quantile only: kept {len(df_CausalFeatures2)} edges with Score >= quantile", flush=True)
    else:
        print(f"  WARNING: Score or Score_quantile column not found, skipping surrogate filtration", flush=True)

elif significance_filter_mode == 'both':
    # Keep edges that pass BOTH tests
    if 'significant_fdr' in df_CausalFeatures2.columns:
        df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['significant_fdr'] == True]
        print(f"  After FDR filter: {len(df_CausalFeatures2)} edges", flush=True)

    if 'Score' in df_CausalFeatures2.columns and 'Score_quantile' in df_CausalFeatures2.columns:
        df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['Score'] >= df_CausalFeatures2['Score_quantile']]
        print(f"  After surrogate quantile filter: {len(df_CausalFeatures2)} edges", flush=True)

    print(f"  Using both filters: kept {len(df_CausalFeatures2)} edges total", flush=True)

print(f"Edges after filtration: {len(df_CausalFeatures2)}\n", flush=True)

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

# Create hybrid DAG with delayed variables
print("Creating hybrid quasi-dynamic DAG with delayed variables...")

# Build enhanced DAG that includes delayed variables
G_dag_enhanced = nx.DiGraph()

# Add all original nodes
for node in G_dag_tmp.nodes():
    G_dag_enhanced.add_node(node)

# Add original edges
for edge in G_dag_tmp.edges():
    G_dag_enhanced.add_edge(edge[0], edge[1])

# Create delayed variables and add edges to enhanced DAG
delayed_variables = {}
edges_to_remove = set()
delayed_edges_to_add = []

# First pass: identify ALL edges with delay > 1 and collect delayed edges to create
# Use df_CausalFeatures2_dag (filtered version) to respect DAG filtering rules
print("\n=== IDENTIFYING DELAYED VARIABLES ===")
for _, row in df_CausalFeatures2_dag.iterrows():
    x1, x2 = row['x1'], row['x2']
    delay = row.get('timeToEffect', 0)

    # Process edges with delay > 1 where source node exists in DAG
    if delay > 1 and x1 in G_dag_enhanced.nodes():
        # Check if direct edge exists or if target node was removed
        edge_exists = G_dag_enhanced.has_edge(x1, x2)
        target_exists = x2 in G_dag_enhanced.nodes()

        if edge_exists:
            # Scenario 1: Direct edge exists - replace with delayed variable
            edges_to_remove.add((x1, x2))
            delayed_var = f"{x1}_shift_t-{int(delay)}"
            delayed_edges_to_add.append((delayed_var, x2, x1, int(delay)))
            delayed_variables[delayed_var] = (x1, int(delay))
            print(f"Will remove direct edge and create delayed: {x1} -> {x2} => {delayed_var} -> {x2} (delay={delay})")
        elif not target_exists:
            # Scenario 3: Target node removed (likely all edges to it are delayed) - recreate with delayed variable
            delayed_var = f"{x1}_shift_t-{int(delay)}"
            delayed_edges_to_add.append((delayed_var, x2, x1, int(delay)))
            delayed_variables[delayed_var] = (x1, int(delay))
            print(f"Target node absent, will create delayed: {delayed_var} -> {x2} (delay={delay})")
        else:
            # Scenario 2: Both nodes exist but edge was removed during cycle-breaking - respect that decision
            print(f"Skipping {x1} -> {x2} (delay={delay}): edge was removed during DAG construction, not recreating")

# Second pass: Remove direct edges that will be replaced
print("\n=== REMOVING DIRECT EDGES ===")
for x1, x2 in edges_to_remove:
    if G_dag_enhanced.has_edge(x1, x2):
        G_dag_enhanced.remove_edge(x1, x2)
        print(f"Removed direct edge: {x1} -> {x2}")

# Third pass: Add delayed variables and their edges
# Only add if target node (x2) has outgoing edges or is a target variable
print("\n=== ADDING DELAYED VARIABLES ===")
added_count = 0
skipped_count = 0
for delayed_var, x2, x1, delay in delayed_edges_to_add:
    # Check if x2 has a path forward (has outgoing edges OR is a target)
    x2_exists = x2 in G_dag_enhanced.nodes()
    x2_has_outgoing = x2_exists and G_dag_enhanced.out_degree(x2) > 0
    x2_is_target = x2 in targetlist

    if x2_has_outgoing or x2_is_target:
        # Ensure x2 (target node) exists
        if not x2_exists:
            G_dag_enhanced.add_node(x2)
            print(f"Re-added node {x2} (needed as target for delayed variable)")

        # Add delayed variable as new node
        if delayed_var not in G_dag_enhanced.nodes():
            G_dag_enhanced.add_node(delayed_var)

        # Add edge from delayed variable to target
        G_dag_enhanced.add_edge(delayed_var, x2)
        print(f"✓ Added delayed variable: {delayed_var} -> {x2} (delay={delay})")
        added_count += 1
    else:
        # Skip: x2 would become a sink node with no path to targets
        print(f"⊗ Skipped {delayed_var} -> {x2}: target node has no outgoing edges and is not a target variable")
        # Don't add to delayed_variables dict since we're not creating it
        if delayed_var in delayed_variables:
            del delayed_variables[delayed_var]
        skipped_count += 1

print(f"\n✓ Replaced {len(edges_to_remove)} direct edges with {added_count} delayed variables ({skipped_count} skipped as sink nodes)")

# Update G_dag_tmp to use the enhanced DAG for all downstream processing
G_dag_tmp = G_dag_enhanced.copy()

# Filter DAG to only include delayed variables that can be created from available data
print("\n=== FILTERING DAG FOR DATA CONSISTENCY ===")
available_original_vars = set(df.columns)  # Variables available in the base data
delayed_vars_to_remove = []

for delayed_var, (original_var, delay) in delayed_variables.items():
    if original_var not in available_original_vars:
        delayed_vars_to_remove.append(delayed_var)
        print(f"!  Removing {delayed_var} from DAG - original variable {original_var} not in data")

# Remove delayed variables that can't be created
for delayed_var in delayed_vars_to_remove:
    if delayed_var in G_dag_tmp.nodes():
        # Remove the delayed variable node and its edges
        G_dag_tmp.remove_node(delayed_var)
        # Remove from delayed_variables dict
        del delayed_variables[delayed_var]
        print(f"X Removed {delayed_var} from DAG and delayed_variables")

# Remove sink nodes (nodes with no outgoing edges) except targets
print("\n=== REMOVING SINK NODES ===")
iteration = 0
total_removed = 0
while True:
    iteration += 1
    sink_nodes = [node for node in G_dag_tmp.nodes()
                  if G_dag_tmp.out_degree(node) == 0 and node not in targetlist]

    if not sink_nodes:
        print(f"✓ No more sinks to remove (iteration {iteration})")
        break

    print(f"Iteration {iteration}: Removing {len(sink_nodes)} sink nodes: {sink_nodes}")
    for sink_node in sink_nodes:
        G_dag_tmp.remove_node(sink_node)
        # Also remove from delayed_variables if it's a delayed variable
        if sink_node in delayed_variables:
            del delayed_variables[sink_node]
        total_removed += 1

print(f"V Removed {total_removed} sink nodes")

edges = G_dag_tmp.edges
DAG = bn.make_DAG(list(edges))
print(f"V Filtered DAG: {len(G_dag_tmp.nodes())} nodes, {len(G_dag_tmp.edges())} edges")

print(f"Enhanced DAG: {len(G_dag_tmp.nodes())} nodes, {len(G_dag_tmp.edges())} edges")
print(f"Delayed variables: {list(delayed_variables.keys())}")

# Remove disconnected fragments - keep only target-connected components
def remove_unconnected_fragments(G, target_nodes):
    """Remove nodes that are not connected to any target node"""
    if not target_nodes or not G.nodes():
        return G
    
    print(f"DEBUG: Looking for targets: {target_nodes}")
    print(f"DEBUG: Available nodes: {list(G.nodes())}")
    
    # Check if any target nodes exist in the graph
    existing_targets = [t for t in target_nodes if t in G.nodes()]
    print(f"DEBUG: Existing targets in graph: {existing_targets}")
    
    if not existing_targets:
        print("WARNING: No target nodes found in graph! Keeping all nodes.")
        return G
    
    # Convert to undirected to find connected components
    G_undirected = G.to_undirected()
    
    # Find all connected components
    connected_components = list(nx.connected_components(G_undirected))
    print(f"DEBUG: Found {len(connected_components)} connected components")
    
    # Find components that contain at least one target node
    target_connected_nodes = set()
    for i, component in enumerate(connected_components):
        has_target = any(target in component for target in existing_targets)
        print(f"DEBUG: Component {i+1}: {len(component)} nodes, has_target: {has_target}")
        if has_target:
            print(f"DEBUG: Keeping component {i+1}: {sorted(component)}")
            target_connected_nodes.update(component)
        else:
            print(f"DEBUG: Removing component {i+1}: {sorted(component)}")
    
    # Create new graph with only target-connected nodes
    G_filtered = G.subgraph(target_connected_nodes).copy()
    
    print(f"Fragment removal: {len(G.nodes())} -> {len(G_filtered.nodes())} nodes")
    print(f"Fragment removal: {len(G.edges())} -> {len(G_filtered.edges())} edges")
    
    return G_filtered

# Apply fragment removal with debugging
print(f"DEBUG: Target list: {targetlist}")
print(f"DEBUG: All nodes in DAG: {list(G_dag_tmp.nodes())}")
print(f"DEBUG: DAG edges: {list(G_dag_tmp.edges())}")

# Apply fragment removal to G_dag_tmp itself (used for BN model)
G_dag_tmp = remove_unconnected_fragments(G_dag_tmp, targetlist)
fig_G_dag = G_dag_tmp  # Dashboard uses the same filtered DAG

# Debug output for Dash app compatibility
print(f"✓ DAG nodes for dashboard: {list(fig_G_dag.nodes())}")
print(f"✓ DAG edges for dashboard: {list(fig_G_dag.edges())}")
print(f"✓ Enhanced DAG includes {len([n for n in fig_G_dag.nodes() if '_shift' in n])} delayed variables")

# Create data with delayed variables
dict_df_cuts = {}
for i in targetlist:
    df_tmp = df_cut.copy()
    
    # Create delayed variable columns - only if original exists
    created_delayed_vars = []
    for delayed_var, (original_var, delay) in delayed_variables.items():
        if original_var in df_tmp.columns:
            df_tmp[delayed_var] = df_tmp[original_var].shift(int(delay))
            created_delayed_vars.append(delayed_var)
            print(f"V Created delayed column: {delayed_var} from {original_var}")
        else:
            print(f"!  MISSING original variable {original_var} - cannot create {delayed_var}")
    
    print(f"V Successfully created {len(created_delayed_vars)} delayed variables: {created_delayed_vars}")
    
    # Apply original shifting logic only to non-delayed variables
    original_cols = [col for col in cols if col not in [dv[0] for dv in delayed_variables.values()]]
    for j in original_cols:
        try:
            s = dict_allLags[(j, i)]
            df_tmp[j] = df_tmp[j].shift(int(s))
        except:
            print("missing interaction")
    
    df_tmp = df_tmp.dropna()
    dict_df_cuts[i] = df_tmp
    
    # Debug output for delayed variables in dataset
    delayed_cols_in_data = [col for col in df_tmp.columns if '_shift' in col]
    expected_delayed_vars = list(delayed_variables.keys())
    print(f"✓ Target {i}: Expected delayed variables: {expected_delayed_vars}")
    print(f"✓ Target {i}: Created delayed columns: {delayed_cols_in_data}")
    print(f"✓ Target {i}: Total columns: {len(df_tmp.columns)}, Rows: {len(df_tmp)}")
    
    # Check for missing delayed variables
    missing_delayed = [var for var in expected_delayed_vars if var not in df_tmp.columns]
    if missing_delayed:
        print(f"!  Target {i}: Missing delayed columns: {missing_delayed}")


dict_acc = {}

# Create debug log file for BN training
bn_debug_file = output_folder + "bn_training_debug.txt"
with open(bn_debug_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("BN TRAINING DEBUG LOG\n")
    f.write("="*60 + "\n\n")

print("\n" + "="*60)
print("STARTING BN TRAINING LOOP")
print("="*60)
for t in targetlist:
    print(f"\n>>> Processing target: {t}")
    print(f">>> dict_df_cuts[{t}] columns: {list(dict_df_cuts[t].columns)}")
    print(f">>> Shifted columns in data: {[c for c in dict_df_cuts[t].columns if '_shift' in c]}")

    dag_nodes = list(DAG['adjmat'].columns)
    print(f">>> DAG nodes (from DAG['adjmat']): {dag_nodes}")
    print(f">>> Shifted nodes in DAG: {[n for n in dag_nodes if '_shift' in n]}")

    # Write to debug file
    with open(bn_debug_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Processing target: {t}\n")
        f.write(f"{'='*60}\n")
        f.write(f"dict_df_cuts[{t}] columns: {list(dict_df_cuts[t].columns)}\n")
        f.write(f"Shifted columns in data: {[c for c in dict_df_cuts[t].columns if '_shift' in c]}\n")
        f.write(f"DAG nodes (from DAG['adjmat']): {dag_nodes}\n")
        f.write(f"Shifted nodes in DAG: {[n for n in dag_nodes if '_shift' in n]}\n")

    # Create a copy of data for BN training
    bn_data = dict_df_cuts[t].copy()

    # Create any missing shifted columns needed for the DAG
    dag_nodes_with_shift = [n for n in dag_nodes if '_shift_t-' in n]
    print(f"\n=== BN TRAINING DEBUG for target {t} ===")
    print(f"DAG nodes with shift: {dag_nodes_with_shift}")

    for node in dag_nodes:
        if node not in bn_data.columns:
            if '_shift_t-' in node:
                # Parse original variable and delay from node name
                parts = node.split('_shift_t-')
                original_var = parts[0]
                delay = int(parts[1])
                if original_var in bn_data.columns:
                    bn_data[node] = bn_data[original_var].shift(delay)
                    print(f"V Created BN data column: {node} from {original_var} (delay={delay})")
                else:
                    print(f"! Cannot create {node}: original {original_var} not in data")

    # Drop NaN rows created by shifting
    bn_data = bn_data.dropna()
    print(f"BN data after creating shifts: {len(bn_data)} rows, {len(bn_data.columns)} columns")

    # Update dict_df_cuts with the new data that includes shifted columns
    dict_df_cuts[t] = bn_data

    available_cols = set(bn_data.columns)
    required_cols = set(dag_nodes)
    missing_cols = required_cols - available_cols
    extra_cols = available_cols - required_cols

    print(f"DAG nodes required: {sorted(dag_nodes)}")
    print(f"Data columns available: {sorted(bn_data.columns)}")
    print(f"Missing columns: {sorted(missing_cols)}")
    print(f"Extra columns: {sorted(extra_cols)}")

    available_dag_cols = [col for col in dag_nodes if col in bn_data.columns]
    if len(available_dag_cols) != len(dag_nodes):
        print(f"!  WARNING: Only {len(available_dag_cols)}/{len(dag_nodes)} DAG columns available in data")
        print(f"Available DAG columns: {available_dag_cols}")
        print("SKIPPING BN training due to missing columns")
        continue  # Skip this target if columns are missing
    else:
        print(f"V All {len(dag_nodes)} DAG columns available - proceeding with BN training")

    # Cross-validation implementation to prevent overfitting
    # Check minimum samples per class for stratification
    min_samples_per_class = dict_df_cuts[t][t].value_counts().min()
    n_folds = min(5, min_samples_per_class)

    # Get all edges from DAG
    edges = list(G_dag_tmp.edges)
    print(f"DEBUG: Total edges in G_dag_tmp: {len(edges)}")

    if n_folds < 2:
        print(f"!  WARNING: Only {min_samples_per_class} samples in smallest class for {t}")
        print("   Using holdout validation instead of cross-validation")

        # Fallback to simple train/test split
        df_cut = dict_df_cuts[t][available_dag_cols].sample(frac=float(bn_training_fraction), random_state=42)
        df_cut_test = dict_df_cuts[t][available_dag_cols].drop(df_cut.index)

        column = t
        df_cut_test = df_cut_test.groupby(column).sample(n=df_cut_test[column].value_counts().min(), random_state=42)

        DAG = bn.make_DAG(list(edges))
        nodes = list(DAG['adjmat'].columns)
        print(f"DEBUG: BN nodes after make_DAG: {nodes}")

        print(f"DEBUG: Training BN with columns: {list(df_cut[nodes].columns)}")
        print(f"DEBUG: Shifted columns in training data: {[c for c in df_cut[nodes].columns if '_shift' in c]}")
        print(f"DEBUG: Training data shape: {df_cut[nodes].shape}")

        # Write to debug file
        with open(bn_debug_file, 'a') as f:
            f.write(f"\n--- HOLDOUT TRAINING ---\n")
            f.write(f"Training BN with columns: {list(df_cut[nodes].columns)}\n")
            f.write(f"Shifted columns in training data: {[c for c in df_cut[nodes].columns if '_shift' in c]}\n")
            f.write(f"Training data shape: {df_cut[nodes].shape}\n")

        DAG_global = bn.parameter_learning.fit(DAG, df_cut[nodes], methodtype='bayes')
        dict_df_cuts[t+"_dag_global"] = DAG_global

        # Debug: Check what edges were learned
        learned_edges = list(DAG_global['model_edges'])
        learned_edge_nodes = [item for sublist in learned_edges for item in sublist]
        learned_edge_nodes = list(set(learned_edge_nodes))
        print(f"DEBUG: Learned edges: {learned_edges}")
        print(f"DEBUG: Shifted nodes in learned edges: {[n for n in learned_edge_nodes if '_shift' in n]}")

        # Write to debug file
        with open(bn_debug_file, 'a') as f:
            f.write(f"Learned edges: {learned_edges}\n")
            f.write(f"Shifted nodes in learned edges: {[n for n in learned_edge_nodes if '_shift' in n]}\n")

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
                    pass
            try:
                q1 = bn.inference.fit(DAG_global, variables=[t], evidence=dict_test)
                all_p.append(q1.df.p[1])
            except:
                all_p.append(0.5)

        df_test = pd.DataFrame()
        df_test['Observed'] = df_cut_test[t].values.tolist()
        df_test['Predicted'] = all_p
        df_test = df_test.astype(float)

        def roundProb(p):
            if p >= probability_cutoff:
                return 1
            else:
                return 0

        df_test['p_binary'] = df_test['Predicted'].apply(roundProb)
        acc = accuracy_score(df_test['Observed'].values, df_test['p_binary'].values)

        print(f"{t} holdout accuracy = {acc:.3f}")
        dict_acc[t] = {'method': 'holdout', 'mean': acc, 'std': None, 'ci_lower': None, 'ci_upper': None}

        # Store the final DAG_global for later use (trained on all training data)
        # This is kept for compatibility with downstream code

    else:
        # Cross-validation implementation
        print(f"✓ Running {n_folds}-fold stratified cross-validation for {t}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accuracies = []
        all_predictions = []
        all_true_labels = []
        all_test_indices = []

        # We'll train a final model on all data after CV for downstream use
        df_all_data = dict_df_cuts[t][available_dag_cols]

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_all_data, df_all_data[t])):
            df_train = df_all_data.iloc[train_idx]
            df_test_fold = df_all_data.iloc[test_idx]

            # Train BN on this fold
            DAG_fold = bn.make_DAG(list(edges))
            nodes_fold = list(DAG_fold['adjmat'].columns)
            DAG_fold = bn.parameter_learning.fit(DAG_fold, df_train[nodes_fold], methodtype='bayes')

            # Get model nodes for prediction
            l_fold = [list(i) for i in DAG_fold['model_edges']]
            model_nodes_fold = [item for sublist in l_fold for item in sublist]
            model_nodes_fold = list(set(model_nodes_fold))

            available_test_nodes_fold = [node for node in model_nodes_fold if node in df_test_fold.columns]

            # Predict on test fold
            fold_preds = []
            for _, row in df_test_fold.iterrows():
                evidence = {col: str(int(row[col])) for col in available_test_nodes_fold if col != t}
                try:
                    q = bn.inference.fit(DAG_fold, variables=[t], evidence=evidence)
                    pred_prob = q.df.p[1] if len(q.df) > 1 else 0.5
                except:
                    pred_prob = 0.5
                fold_preds.append(pred_prob)

            # Calculate fold accuracy
            pred_binary = [1 if p >= probability_cutoff else 0 for p in fold_preds]
            true_labels = df_test_fold[t].astype(int).tolist()
            fold_acc = sum([1 for p, t_label in zip(pred_binary, true_labels) if p == t_label]) / len(true_labels)
            fold_accuracies.append(fold_acc)

            all_predictions.extend(fold_preds)
            all_true_labels.extend(true_labels)
            all_test_indices.extend(test_idx)

            print(f"  Fold {fold_idx+1}/{n_folds}: accuracy = {fold_acc:.3f}")

        # Calculate CV statistics
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        ci_lower = mean_acc - 1.96 * std_acc / np.sqrt(len(fold_accuracies))
        ci_upper = mean_acc + 1.96 * std_acc / np.sqrt(len(fold_accuracies))

        print(f"\n✓ {t} CV accuracy = {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        dict_acc[t] = {
            'method': 'cross_validation',
            'mean': mean_acc,
            'std': std_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fold_accuracies': fold_accuracies
        }

        # Train final model on ALL data for downstream use and visualization
        DAG = bn.make_DAG(list(edges))
        nodes = list(DAG['adjmat'].columns)

        print(f"DEBUG CV: Training BN with columns: {list(df_all_data[nodes].columns)}")
        print(f"DEBUG CV: Shifted columns in training data: {[c for c in df_all_data[nodes].columns if '_shift' in c]}")
        print(f"DEBUG CV: Training data shape: {df_all_data[nodes].shape}")

        # Write to debug file
        with open(bn_debug_file, 'a') as f:
            f.write(f"\n--- CV TRAINING ---\n")
            f.write(f"Training BN with columns: {list(df_all_data[nodes].columns)}\n")
            f.write(f"Shifted columns in training data: {[c for c in df_all_data[nodes].columns if '_shift' in c]}\n")
            f.write(f"Training data shape: {df_all_data[nodes].shape}\n")

        DAG_global = bn.parameter_learning.fit(DAG, df_all_data[nodes], methodtype='bayes')
        dict_df_cuts[t+"_dag_global"] = DAG_global

        # Debug: Check what edges were learned
        learned_edges = list(DAG_global['model_edges'])
        learned_edge_nodes = [item for sublist in learned_edges for item in sublist]
        learned_edge_nodes = list(set(learned_edge_nodes))
        print(f"DEBUG CV: Learned edges: {learned_edges}")
        print(f"DEBUG CV: Shifted nodes in learned edges: {[n for n in learned_edge_nodes if '_shift' in n]}")

        # Write to debug file
        with open(bn_debug_file, 'a') as f:
            f.write(f"Learned edges: {learned_edges}\n")
            f.write(f"Shifted nodes in learned edges: {[n for n in learned_edge_nodes if '_shift' in n]}\n")

        DAG_global_learned = bn.structure_learning.fit(df_all_data[nodes])
        dict_df_cuts[t+"_dag_global_learned"] = DAG_global_learned

        # Create df_test using CV predictions for visualization
        df_test = pd.DataFrame()
        df_test['Observed'] = [all_true_labels[all_test_indices.index(i)] for i in sorted(all_test_indices)]
        df_test['Predicted'] = [all_predictions[all_test_indices.index(i)] for i in sorted(all_test_indices)]
        df_test = df_test.astype(float)

        def roundProb(p):
            if p >= probability_cutoff:
                return 1
            else:
                return 0

        df_test['p_binary'] = df_test['Predicted'].apply(roundProb)

    # Common plotting code for both holdout and CV
    plt.figure(figsize=(6, 10))
    ax = df_test.reset_index().plot(kind="scatter", s=30, x="index", y="Predicted", c="orange", figsize=(6, 10))
    df_test.reset_index().plot(kind="scatter", x="index", y="Observed", secondary_y=False, ax=ax)
    plt.ylabel('Probability', fontsize=24)
    plt.xlabel('Test samples', fontsize=24)
    plt.savefig(output_folder + 'BN_model_validation.png', bbox_inches='tight', transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.boxplot(x=df_test["Observed"], y=df_test["Predicted"], ax=ax, boxprops={"facecolor": (.4, .6, .8, .5)})
    ax.set_xlabel("Observed", fontsize=24)
    ax.set_ylabel("Predicted", fontsize=24)
    plt.tight_layout()
    plt.savefig(output_folder + 'BN_model_results.png', bbox_inches='tight', transparent=True)
    plt.close()

    cm = confusion_matrix(df_test['Observed'].values, df_test['p_binary'].values)
    cm_transposed = cm.T

    plt.figure(figsize=(6, 10))
    sns.set(font_scale=2)
    sns.heatmap(cm_transposed, annot=True, cmap="Blues", fmt="d")
    plt.xlabel('Observed', fontsize=24)
    plt.ylabel('Predicted', fontsize=24)
    plt.savefig(output_folder + 'BN_model_confusionMatrix.png', bbox_inches='tight', transparent=True)
    plt.close()

    AllNodes = list(DAG_global['adjmat'].columns)
    
    AllEdges = edges

    g = pydot.Dot()

    for node in AllNodes:
        if node in targetlist:
            g.add_node(pydot.Node(node, color='cyan', style='filled'))
        elif '_shift' in node:
            # Delayed variables in purple with dashed border
            g.add_node(pydot.Node(node, color='mediumpurple', style='filled', penwidth='2', peripheries='2'))
        else:
            g.add_node(pydot.Node(node, color='orange', style='filled'))

    for i in AllEdges:
        # Different edge styles for delayed vs original edges
        if '_shift' in i[0]:
            # Delayed variable edges are dashed
            g.add_edge(pydot.Edge(i[0], i[1], color='purple', style='dashed', penwidth='2'))
        else:
            g.add_edge(pydot.Edge(i[0], i[1], color='black'))
            
    # Much taller than wide for dashboard with high quality
    g.set('size', '"12,4!"')  # Width x Height - much taller than wide
    g.set('dpi', '300')  # High quality
    g.set('rankdir', 'TB')  # Top to bottom layout
    g.set('ratio', 'compress')  # Compress to fit size
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
    # Use AllNodes (from adjmat.columns) which includes all structural nodes
    for j, valj in enumerate(path[:-1]):
        try:
            if valj in AllNodes:
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

    # Use adjmat.columns for all nodes (includes nodes even if some edges didn't learn properly)
    AllNodes = list(DAG_global['adjmat'].columns)

    # Also get nodes from model_edges for comparison
    edge_nodes = [item for sublist in DAG_global['model_edges'] for item in sublist]
    edge_nodes = list(set(edge_nodes))

    print(f"\n=== SENSITIVITY ANALYSIS DEBUG for {t} ===")
    print(f"Nodes in adjmat (structure): {AllNodes}")
    print(f"Nodes in model_edges (learned): {edge_nodes}")
    print(f"Shifted nodes in adjmat: {[n for n in AllNodes if '_shift' in n]}")
    print(f"Shifted nodes in model_edges: {[n for n in edge_nodes if '_shift' in n]}")

    # Check for nodes in structure but not in learned edges
    missing_in_edges = set(AllNodes) - set(edge_nodes)
    if missing_in_edges:
        print(f"!  WARNING: Nodes in structure but NOT in learned edges: {missing_in_edges}")

    # Write to debug file
    with open(bn_debug_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"SENSITIVITY ANALYSIS for {t}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Nodes in adjmat (structure): {AllNodes}\n")
        f.write(f"Nodes in model_edges (learned): {edge_nodes}\n")
        f.write(f"Shifted nodes in adjmat: {[n for n in AllNodes if '_shift' in n]}\n")
        f.write(f"Shifted nodes in model_edges: {[n for n in edge_nodes if '_shift' in n]}\n")
        if missing_in_edges:
            f.write(f"WARNING: Nodes in structure but NOT in learned edges: {missing_in_edges}\n")

    dictBounds = {}
    dict_NodesUniqueValues = {}

    for j in dict_df_cuts[t].columns:
        unq = list(dict_df_cuts[t][j].unique())
        unq = [int(k) for k in unq]

        dictBounds[j] = discreteBounds(unq)
        dict_NodesUniqueValues[j] = [str(u) for u in unq]


    trained_model_nodes = list(DAG_global['adjmat'].columns)

    # Debug: Show what nodes are in the trained model
    shifted_in_model = [n for n in trained_model_nodes if '_shift' in n]
    print(f"DEBUG: Trained model nodes: {trained_model_nodes}")
    print(f"DEBUG: Shifted variables in model: {shifted_in_model}")
    print(f"DEBUG: dictBounds keys: {list(dictBounds.keys())}")

    # Ensure dictBounds and dict_NodesUniqueValues have entries for all trained model nodes (including shifted)
    for node in trained_model_nodes:
        if node not in dictBounds:
            # For shifted variables, get bounds from the original variable
            if '_shift_t-' in node:
                original_var = node.split('_shift_t-')[0]
                if original_var in dictBounds:
                    dictBounds[node] = dictBounds[original_var]
                    dict_NodesUniqueValues[node] = dict_NodesUniqueValues.get(original_var, ['0', '1', '2'])
                    print(f"DEBUG: Added dictBounds[{node}] from {original_var}")
                else:
                    # Default bounds if original not found
                    dictBounds[node] = [(0, 2)]
                    dict_NodesUniqueValues[node] = ['0', '1', '2']
                    print(f"DEBUG: Added default dictBounds[{node}]")
            else:
                dictBounds[node] = [(0, 2)]
                dict_NodesUniqueValues[node] = ['0', '1', '2']
                print(f"DEBUG: Added default dictBounds[{node}]")

    # Construct path using trained model node names
    path = [node for node in trained_model_nodes if node != t]
    path = path + [t]

    print(f"DEBUG: Path for scenarios: {path}")

    bounds = setBounds(path, dictBounds)

    
    # Use dict_df_cuts[t] which includes shifted variables, not df_cut
    data_for_scenarios = dict_df_cuts[t].copy()

    # Create any missing shifted columns from original variables
    missing_cols = set(path) - set(data_for_scenarios.columns)
    if missing_cols:
        print(f"DEBUG: Data missing columns: {missing_cols}")
        for col in missing_cols:
            if '_shift_t-' in col:
                original_var = col.split('_shift_t-')[0]
                delay = int(col.split('_shift_t-')[1])
                if original_var in data_for_scenarios.columns:
                    data_for_scenarios[col] = data_for_scenarios[original_var].shift(delay)
                    print(f"DEBUG: Created shifted column {col} from {original_var} with delay {delay}")

        # Drop NaN rows created by shifting
        data_for_scenarios = data_for_scenarios.dropna()
        print(f"DEBUG: After creating shifted columns, data has {len(data_for_scenarios)} rows")

    # Check again for any still-missing columns
    still_missing = set(path) - set(data_for_scenarios.columns)
    if still_missing:
        print(f"WARNING: Still missing columns after shift creation: {still_missing}")
        actual_path = [col for col in path if col in data_for_scenarios.columns]
        if not actual_path:
            print(f"ERROR: No path columns available in data")
            listOfRandomVecs = []
        else:
            if t in actual_path:
                actual_path.remove(t)
            actual_path = actual_path + [t]
            listOfRandomVecs = data_for_scenarios[actual_path].astype(int).values.tolist()
    else:
        actual_path = path
        listOfRandomVecs = data_for_scenarios[path].astype(int).values.tolist()

    print(f"Sensitivity analysis using {len(actual_path)-1} variables: {actual_path[:-1]}")

    # Write to debug file
    with open(bn_debug_file, 'a') as f:
        f.write(f"\nSensitivity analysis using {len(actual_path)-1} variables: {actual_path[:-1]}\n")
        f.write(f"Full actual_path: {actual_path}\n")

    # max
    # Pre-compute nodes from model_edges for this target's DAG_global
    l_edges = [list(i) for i in DAG_global['model_edges']]
    model_edge_nodes = [item for sublist in l_edges for item in sublist]
    model_edge_nodes = list(set(model_edge_nodes))

    print(f"DEBUG: model_edge_nodes: {model_edge_nodes}")
    print(f"DEBUG: Shifted in model_edge_nodes: {[n for n in model_edge_nodes if '_shift' in n]}")
    print(f"DEBUG: actual_path: {actual_path}")

    # Write to debug file
    with open(bn_debug_file, 'a') as f:
        f.write(f"model_edge_nodes: {model_edge_nodes}\n")
        f.write(f"Shifted in model_edge_nodes: {[n for n in model_edge_nodes if '_shift' in n]}\n")

    for row in listOfRandomVecs:
        try:
            result = f_max(row[:-1])
            v = [round(i) for i in row[:-1]]
            print(str(v))
            dict_evidence = {}
            # Use actual_path for consistent indexing with v
            # Include ALL nodes from the structural model (AllNodes = adjmat.columns)
            for idx, valj in enumerate(actual_path[:-1]):
                try:
                    # Include variable if it's in the structural model (AllNodes)
                    # The inference will use it if it has learned CPDs
                    if valj in AllNodes:
                        dict_evidence[valj] = str(v[idx])
                except:
                    print()

            for j in targetlist:
                try:
                    del dict_evidence[j]
                except:
                    print()
            print(dict_evidence)
            res_sub_max.append([path[-1], dict_evidence, result])
        except Exception as e:
            print(f"Scenario analysis error: {e}")


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


# FIGURE - Update cols to include delayed variables
cols = df_de_max_vecs.columns
cols = [i for i in cols if not i in ['Score', 'y']]

# Ensure delayed variables are included in cols for plotting
all_dag_cols = list(G_dag_tmp.nodes())
plot_cols = [col for col in all_dag_cols if col not in targetlist]
if plot_cols:
    cols = plot_cols
    print(f"Updated cols to include delayed variables: {cols}")

# Filter cols to only include columns that exist in df_de_max_vecs
available_cols = [col for col in cols if col in df_de_max_vecs.columns]
if not available_cols:
    # Fallback to original cols if none of the updated cols exist
    available_cols = [col for col in df_de_max_vecs.columns if col not in ['Score', 'y']]
print(f"Available cols for filtering: {available_cols}")
df_de_max_vecs_filtered = df_de_max_vecs[available_cols].astype(float)

df_CausalFeatures2 = df_CausalFeatures2.dropna()
df_de_max = df_de_max_vecs.copy()  # Keep original with Score column
df_de_min = df_de_min_vecs.copy()
plt.close()

high_scenarios = []
low_scenarios = []

if len(df_de_max_vecs) > 0:
    for _, row in df_de_max_vecs.iterrows():
        scenario = {col: int(row[col]) for col in available_cols if col in row}
        high_scenarios.append(scenario)

if len(df_de_min_vecs) > 0:
    for _, row in df_de_min_vecs.iterrows():
        scenario = {col: int(row[col]) for col in available_cols if col in row}
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

# Add completion marker
completion_file = output_folder + "bn_frequencies_complete.txt"
with open(completion_file, 'w') as f:
    f.write("BN frequency calculations completed successfully\n")
    f.write(f"High scenarios: {len(high_scenarios)}\n") 
    f.write(f"Low scenarios: {len(low_scenarios)}\n")
    f.write(f"High frequencies: {len(high_frequencies)}\n")
    f.write(f"Low frequencies: {len(low_frequencies)}\n")

# Add progress marker before plot generation
progress_file = output_folder + "bn_plot_progress.txt"
with open(progress_file, 'w') as f:
    f.write("Starting plot generation phase\n")

for t in targetlist:
    
    if len(df_de_max) > 0:

        df_mean_max = df_de_max_vecs[available_cols].astype(int).mean()
        l_max = df_mean_max.reset_index().values.tolist()
        print(f"DEBUG: MEAN MAX scenario values: {df_mean_max.to_dict()}")
    else:
        l_max = [(col, 1) for col in available_cols]  
    

    values_list_max = [val for _, val in l_max]
    variable_names_max = [var for var, _ in l_max]
    d_max = generate_distinct_colors(values_list_max, variable_names_max, "max")

    
    g_max = pydot.Dot()
    
    edgesL_ = [i[0] for i in edges]
    edgesR_ = [i[1] for i in edges]
    edges_ = [(edgesL_[i], edgesR_[i]) for i in range(0, len(edgesL_))]

    for node in cols:
        if node not in t:
            # Get color using full node name (including shift suffix)
            node_color = d_max.get(node, '0.5 1.0 1.0')

            if '_shift' in node:
                # Delayed variables with special styling (double border)
                nd = pydot.Node(node,
                                style='filled',
                                fontsize="20pt",
                                fillcolor=node_color,
                                penwidth='2',
                                peripheries='2')
            else:
                nd = pydot.Node(node,
                                style='filled',
                                fontsize="20pt",
                                fillcolor=node_color)
            g_max.add_node(nd)

    for c, i in enumerate(edges):
        # Handle delayed variable edges differently
        if '_shift' in i[0]:
            # For delayed variables, extract original timeToEffect from variable name
            parts = i[0].split('_shift')
            if len(parts) == 2:
                lbl = parts[1]  # Use the shift amount as label
                is_direct = 'purple'
                edge_style = 'dashed'
            else:
                lbl = 'delayed'
                is_direct = 'purple'
                edge_style = 'dashed'
        else:
            # Original edge handling
            matching_rows = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) & (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()
            if len(matching_rows) > 0:
                lbl = matching_rows[0]
            else:
                lbl = 0  
            if (lbl >= 0) and (lbl <= 2):
                is_direct = 'black'
            else:
                is_direct = 'gray'
            edge_style = 'filled'

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
                              style=edge_style,
                              #label=lbl,
                              fontsize="20pt"))
    
    # Much taller than wide for dashboard with high quality
    g_max.set('size', '"12,4!"')  # Width x Height - much taller than wide
    g_max.set('dpi', '300')  # High quality
    g_max.set('rankdir', 'TB')  # Top to bottom layout
    g_max.set('ratio', 'compress')  # Compress to fit size
    g_max.write_png(output_folder+"CausalDAG_NET_MEAN_MAX.png")

    if len(df_de_min) > 0:
        # Calculate MEAN of all low probability scenarios
        df_mean_min = df_de_min_vecs[available_cols].astype(int).mean()
        l_min = df_mean_min.reset_index().values.tolist()
    else:
        l_min = [(col, 1) for col in available_cols] 
        
    values_list_min = [val for _, val in l_min]
    variable_names_min = [var for var, _ in l_min]
    d_min = generate_distinct_colors(values_list_min, variable_names_min, "min")

    g_min = pydot.Dot()

    for node in cols:
        if node not in t:
            # Get color using full node name (including shift suffix)
            node_color = d_min.get(node, '0.5 1.0 1.0')

            if '_shift' in node:
                # Delayed variables with special styling (double border)
                nd = pydot.Node(node,
                                style='filled',
                                fontsize="20pt",
                                fillcolor=node_color,
                                penwidth='2',
                                peripheries='2')
            else:
                nd = pydot.Node(node,
                                style='filled',
                                fontsize="20pt",
                                fillcolor=node_color)
            g_min.add_node(nd)

    for c, i in enumerate(edges):
        # Handle delayed variable edges differently
        if '_shift' in i[0]:
            # For delayed variables, extract original timeToEffect from variable name
            parts = i[0].split('_shift')
            if len(parts) == 2:
                lbl = parts[1]  # Use the shift amount as label
                is_direct = 'purple'
                edge_style = 'dashed'
            else:
                lbl = 'delayed'
                is_direct = 'purple'
                edge_style = 'dashed'
        else:
            # Original edge handling
            matching_rows = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) & (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()
            if len(matching_rows) > 0:
                lbl = matching_rows[0]
            else:
                lbl = 0         
            if (lbl >= 0) and (lbl <= 2):
                is_direct = 'black'
            else:
                is_direct = 'gray'
            edge_style = 'filled'

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
                              style=edge_style,
                              #label=lbl,
                              fontsize="20pt"))
    
    # Much taller than wide for dashboard with high quality
    g_min.set('size', '"12,4!"')  # Width x Height - much taller than wide
    g_min.set('dpi', '300')  # High quality
    g_min.set('rankdir', 'TB')  # Top to bottom layout
    g_min.set('ratio', 'compress')  # Compress to fit size
    g_min.write_png(output_folder+"CausalDAG_NET_MEAN_MIN.png")


    # Generate individual scenario PNGs with error handling
    print(f"\nGenerating {len(high_scenarios)} high probability scenario graphs...")
    for i, scenario in enumerate(high_scenarios):
        try:
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(high_scenarios)} high scenarios completed")

            l_individual = [(col, scenario[col]) for col in available_cols if col in scenario]
            values_list = [val for _, val in l_individual]
            variable_names = [var for var, _ in l_individual]
            d_individual = generate_distinct_colors(values_list, variable_names, "max")

            g_individual = pydot.Dot()

            for node in cols:
                if node not in t:
                    # Get color using full node name (including shift suffix)
                    node_color = d_individual.get(node, '0.5 1.0 1.0')

                    nd = pydot.Node(node,
                                    style='filled',
                                    fontsize="20pt",
                                    fillcolor=node_color)
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
                                      #label=lbl,
                                      fontsize="20pt"))

            # Much taller than wide for dashboard with high quality
            g_individual.set('size', '"12,4!"')  # Width x Height - much taller than wide
            g_individual.set('dpi', '300')  # High quality
            g_individual.set('rankdir', 'TB')  # Top to bottom layout
            g_individual.set('ratio', 'compress')  # Compress to fit size
            g_individual.write_png(output_folder+f"scenario_high_{i:03d}.png")

        except Exception as e:
            print(f"  !  Error creating scenario_high_{i:03d}.png: {e}")
            continue

    print(f"✓ Completed {len(high_scenarios)} high probability scenario graphs")


    # Generate low probability scenario PNGs with error handling
    print(f"\nGenerating {len(low_scenarios)} low probability scenario graphs...")
    for i, scenario in enumerate(low_scenarios):
        try:
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(low_scenarios)} low scenarios completed")

            l_individual = [(col, scenario[col]) for col in available_cols if col in scenario]
            values_list = [val for _, val in l_individual]
            variable_names = [var for var, _ in l_individual]
            d_individual = generate_distinct_colors(values_list, variable_names, "min")

            g_individual = pydot.Dot()

            for node in cols:
                if node not in t:
                    # Get color using full node name (including shift suffix)
                    node_color = d_individual.get(node, '0.5 1.0 1.0')

                    nd = pydot.Node(node,
                                    style='filled',
                                    fontsize="20pt",
                                    fillcolor=node_color)
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
                                      #label=lbl,
                                      fontsize="20pt"))

            # Much taller than wide for dashboard with high quality
            g_individual.set('size', '"12,4!"')  # Width x Height - much taller than wide
            g_individual.set('dpi', '300')  # High quality
            g_individual.set('rankdir', 'TB')  # Top to bottom layout
            g_individual.set('ratio', 'compress')  # Compress to fit size
            g_individual.write_png(output_folder+f"scenario_low_{i:03d}.png")

        except Exception as e:
            print(f"  !  Error creating scenario_low_{i:03d}.png: {e}")
            continue

    print(f"✓ Completed {len(low_scenarios)} low probability scenario graphs")

# Create progress marker
print("\n" + "="*60)
print("CONTINUING TO HEATMAPS AND FINAL OUTPUTS")
print("="*60)

try:
    df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
    df_de_max = df_de_max.drop_duplicates()
except Exception as e:
    print(f"!  Warning: Could not create df_de_max DataFrame: {e}")
    df_de_max = pd.DataFrame()



#######Figures#########
print("\nGenerating final heatmap visualizations...")
try:
    for t in targetlist:
        try:
            DAG_global_learned = dict_df_cuts[t+"_dag_global_learned"]
            learned_dags_djmat = DAG_global_learned['adjmat']*1

            plt.figure(figsize=(6, 10))  # Taller format for dashboard
            g = sns.clustermap(learned_dags_djmat, cbar=False, col_cluster=False, row_cluster=False, linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)
            g.ax_row_dendrogram.set_visible(False)
            g.ax_col_dendrogram.set_visible(False)



            g.ax_heatmap.set_xlabel("Target", fontsize=24, labelpad=20)
            g.ax_heatmap.set_ylabel("Source", fontsize=24, labelpad=20)


            if g.cax is not None:
                g.cax.set_visible(False)
            plt.savefig(output_folder + 'learned_fromCCMfeatures_dag.png', bbox_inches='tight', transparent=True)
            plt.close()
            print(f"  ✓ Created learned_fromCCMfeatures_dag.png")

            #######


            dict_df_cuts[t+"_dag_global"]
            ccm_dags_djmat = DAG_global['adjmat']*1

            plt.figure(figsize=(6, 10))  # Taller format for dashboard

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
            print(f"  ✓ Created ccm_dag.png")

        except Exception as e:
            print(f"  !  Error creating heatmaps for target {t}: {e}")
            continue

        #######
except Exception as e:
    print(f"!  Error in heatmap generation section: {e}")

try:
    df_CausalFeatures2 = pd.read_csv(output_folder+"/Surr_filtered.csv")
except:
    try:
        df_CausalFeatures2 = pd.read_csv(output_folder+"/tmp/CCM_ECCM_curated.csv")
    except:
        try:
            df_CausalFeatures2 = pd.read_csv(output_folder+"/CCM_ECCM_curated.csv")
        except Exception as e:
            print(f"!  Warning: Could not load CausalFeatures2 CSV: {e}")
            df_CausalFeatures2 = pd.DataFrame()

try:
    if not df_CausalFeatures2.empty:
        ccm_eccm = df_CausalFeatures2.pivot(index='x1', columns='x2', values='Score')
    else:
        ccm_eccm = pd.DataFrame()
except Exception as e:
    print(f"!  Warning: Could not create ccm_eccm pivot: {e}")
    ccm_eccm = pd.DataFrame()



try:
    if not ccm_eccm.empty:
        plt.rcParams['figure.figsize'] = (6, 10)  # Taller format for dashboard
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
        print(f"  ✓ Created ccm_eccm.png")
    else:
        print(f"  !  Skipping ccm_eccm.png (empty data)")
except Exception as e:
    print(f"!  Error creating ccm_eccm plot: {e}")

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


plt.figure(figsize=(6, 10))  # Taller format for dashboard
sum_mean_contributions.sort_values().plot(kind="bar", capsize=4, color='blue')

plt.xlabel('Input Variable', fontsize=16)
plt.ylabel('Sum of |Mean Contribution|', fontsize=16)

plt.grid(False)
plt.gca().set_facecolor('white')
plt.savefig(output_folder + "sensitivity_barplot.png", bbox_inches='tight', dpi=600)
plt.close()

try:
    bn.save(DAG_global, filepath=output_folder+'bnlearn_model', overwrite=True)
    print("✓ Saved bnlearn model")
except Exception as e:
    print(f"!  Warning: Could not save bnlearn model: {e}")


dict_model_essentials = {}
dict_model_essentials["nodes"] = path[:-1]
dict_model_essentials["target"] = path[-1]

# Get accuracy from dict_acc (handles both old and new format)
target_key = path[-1]
if target_key in dict_acc:
    acc_data = dict_acc[target_key]
    if isinstance(acc_data, dict):
        # New CV format
        dict_model_essentials["accuracy"] = acc_data['mean']
        dict_model_essentials["accuracy_std"] = acc_data.get('std', None)
        dict_model_essentials["accuracy_ci_lower"] = acc_data.get('ci_lower', None)
        dict_model_essentials["accuracy_ci_upper"] = acc_data.get('ci_upper', None)
    else:
        # Old format (simple float)
        dict_model_essentials["accuracy"] = acc_data
else:
    dict_model_essentials["accuracy"] = None
    print(f"!  Warning: No accuracy found for target {target_key}")

# ROC AUC might not be set
dict_model_essentials["roc_auc"] = rocauc if 'rocauc' in locals() else -1

try:
    test_evidence = {node: '1' for node in dict_model_essentials['nodes'][:2]}
    print(f"Testing inference with saved node names: {test_evidence}")
    test_q = bn.inference.fit(DAG_global, variables=[dict_model_essentials['target']], evidence=test_evidence)
    print(f"Inference test SUCCESS with saved names")
except Exception as e:
    print(f"Inference test FAILED with saved names: {e}")

try:
    with open(output_folder+'dict_model_essentials.pickle', 'wb') as handle:
        pickle.dump(dict_model_essentials, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved dict_model_essentials.pickle")
except Exception as e:
    print(f"!  Warning: Could not save dict_model_essentials.pickle: {e}")

try:
    with open(output_folder+'bounds.pickle', 'wb') as handle:
        pickle.dump(dictBounds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved bounds.pickle")
except Exception as e:
    print(f"!  Warning: Could not save bounds.pickle: {e}")

# Save accuracy results with confidence intervals
print("\n" + "="*70)
print("SAVING ACCURACY RESULTS WITH CONFIDENCE INTERVALS")
print("="*70)

acc_results = []
print(f"Processing accuracy data for {len(dict_acc)} target(s)...")

for target, acc_data in dict_acc.items():
    print(f"  - {target}: {type(acc_data).__name__}")
    if isinstance(acc_data, dict):
        acc_results.append({
            'target': target,
            'method': acc_data['method'],
            'accuracy_mean': round(acc_data['mean'], 4),
            'accuracy_std': round(acc_data.get('std', 0), 4) if acc_data.get('std') is not None else None,
            'ci_lower': round(acc_data.get('ci_lower', 0), 4) if acc_data.get('ci_lower') is not None else None,
            'ci_upper': round(acc_data.get('ci_upper', 0), 4) if acc_data.get('ci_upper') is not None else None
        })
    else:
        # Legacy format (simple float)
        acc_results.append({
            'target': target,
            'method': 'legacy',
            'accuracy_mean': round(acc_data, 4),
            'accuracy_std': None,
            'ci_lower': None,
            'ci_upper': None
        })

if acc_results:
    try:
        df_accuracies = pd.DataFrame(acc_results)
        csv_path = output_folder + 'model_accuracies_with_ci.csv'
        df_accuracies.to_csv(csv_path, index=False)
        print(f"\n✓✓✓ SUCCESS: Model accuracies saved to: {csv_path}")
        print(f"    File size: {len(open(csv_path).read())} bytes")

        # Print summary
        print("\n=== MODEL ACCURACY SUMMARY ===")
        for result in acc_results:
            if result['method'] == 'cross_validation':
                print(f"  {result['target']}: {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f} "
                      f"(95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}])")
            else:
                print(f"  {result['target']}: {result['accuracy_mean']:.3f} ({result['method']})")
    except Exception as e:
        print(f"X ERROR: Could not save model_accuracies_with_ci.csv: {e}")
        import traceback
        traceback.print_exc()
else:
    print("!  WARNING: No accuracy results to save (acc_results is empty)")

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
