#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of CCM/ECCM results 
@author: ofir
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import networkx as nx
import os
import argparse
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# PC Algorithm imports
try:
    from scipy.stats import pearsonr, spearmanr
    from itertools import combinations
    PC_AVAILABLE = True
except ImportError:
    PC_AVAILABLE = False

# Transfer Entropy imports
try:
    from jpype import startJVM, shutdownJVM, JClass
    import jpype
    JIDT_AVAILABLE = False
    # Try to import JIDT for Transfer Entropy
    try:
        # Check if JVM is already running
        if not jpype.isJVMStarted():
            # You'll need to download infodynamics.jar from https://github.com/jlizier/jidt
            startJVM(classpath="infodynamics.jar")
        JIDT_AVAILABLE = True
    except:
        pass
except ImportError:
    JIDT_AVAILABLE = False

# Alternative TE implementation using entropy estimators
try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_MI_AVAILABLE = True
except ImportError:
    SKLEARN_MI_AVAILABLE = False


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def make_stationary(column):
    """Make time series stationary if needed."""
    adf_result = adfuller(column)
    p_value = adf_result[1]
    if p_value >= 0.05:
        diff_column = column.diff()
        return diff_column
    else:
        return column


def preprocess_data(df, z_score_threshold=3.0, resample_freq='D'):
    """Preprocess data in the same way as CCM/ECCM pipeline."""
    
    # Handle date column
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    except:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        except:
            pass
    
    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Normalize 0-1
    df_normalized = pd.DataFrame(index=df.index)
    scalers_dict = {}
    for col in df.columns:
        scaler = MinMaxScaler((0, 1))
        scaled_data = scaler.fit_transform(df[col].values.reshape(-1, 1))
        df_normalized[col] = [j[0] for j in scaled_data]
        scalers_dict[col] = scaler
    
    # Remove outliers
    for col in df_normalized.columns:
        mask = (np.abs(stats.zscore(df_normalized[col])) > z_score_threshold)
        df_normalized[col] = df_normalized[col].mask(mask).interpolate()
    
    # Resample if needed
    if hasattr(df_normalized.index, 'freq') or pd.api.types.is_datetime64_any_dtype(df_normalized.index):
        df_normalized = df_normalized.resample(resample_freq).interpolate(method='linear')
    
    # Make stationary
    for col in df_normalized.columns:
        try:
            df_normalized[col] = make_stationary(df_normalized[col])
        except:
            pass
    
    df_normalized = df_normalized.dropna()
    
    return df_normalized, scalers_dict


def partial_correlation(x, y, z_vars, data):
    """Calculate partial correlation between x and y given z_vars"""
    if len(z_vars) == 0:
        # Simple correlation if no conditioning variables
        return pearsonr(data[x], data[y])
    
    try:
        # Use linear regression to compute partial correlation
        from sklearn.linear_model import LinearRegression
        
        # Regress x on z_vars
        reg_x = LinearRegression().fit(data[z_vars], data[x])
        residual_x = data[x] - reg_x.predict(data[z_vars])
        
        # Regress y on z_vars  
        reg_y = LinearRegression().fit(data[z_vars], data[y])
        residual_y = data[y] - reg_y.predict(data[z_vars])
        
        # Correlation of residuals is partial correlation
        return pearsonr(residual_x, residual_y)
    except:
        return 0.0, 1.0  # Return no correlation if calculation fails


def run_pc_algorithm_efficient(df, alpha=0.05, max_lag=12):
    """Run efficient PC algorithm focusing on direct causal relationships"""
    
    if not PC_AVAILABLE:
        print("WARNING: Required libraries not available for PC algorithm.")
        return None, None
    
    var_names = list(df.columns)
    edges = []
    
    print(f"PC Algorithm: Analyzing {len(var_names)} variables with lags up to {max_lag}")
    
    # Test direct pairwise relationships at each lag
    for source_var in var_names:
        for target_var in var_names:
            if source_var == target_var:
                continue
            
            # Test each lag individually
            for lag in range(1, max_lag + 1):
                try:
                    # Create lagged source and current target
                    source_lagged = df[source_var].shift(lag).dropna()
                    target_current = df[target_var].iloc[lag:len(source_lagged)+lag]
                    
                    if len(source_lagged) != len(target_current) or len(source_lagged) < 50:
                        continue
                    
                    # Simple correlation test (PC algorithm with conditioning set size 0)
                    corr, p_val = pearsonr(source_lagged, target_current)
                    
                    # Also test with target's own past as conditioning (basic PC)
                    target_lagged = df[target_var].shift(lag).dropna()
                    if len(target_lagged) == len(target_current):
                        # Partial correlation conditioning on target's past
                        try:
                            from sklearn.linear_model import LinearRegression
                            
                            # Regress target_current on target_lagged
                            reg_target = LinearRegression().fit(target_lagged.values.reshape(-1, 1), target_current.values)
                            residual_target = target_current.values - reg_target.predict(target_lagged.values.reshape(-1, 1))
                            
                            # Regress source_lagged on target_lagged  
                            reg_source = LinearRegression().fit(target_lagged.values.reshape(-1, 1), source_lagged.values)
                            residual_source = source_lagged.values - reg_source.predict(target_lagged.values.reshape(-1, 1))
                            
                            # Correlation between residuals
                            partial_corr, partial_p = pearsonr(residual_source, residual_target)
                            
                            # Use partial correlation results
                            corr = partial_corr
                            p_val = partial_p
                        except:
                            # Fall back to simple correlation
                            pass
                    
                    if p_val < alpha and abs(corr) > 0.1:  # Significant relationship
                        edges.append({
                            'x1': source_var,  # Source 
                            'x2': target_var,  # Target
                            'lag': lag,
                            'correlation': corr,
                            'p_value': p_val,
                            'abs_correlation': abs(corr),
                            'method': 'PC'
                        })
                        
                except Exception as e:
                    continue
    
    return edges, None

def run_pc_algorithm(df, alpha=0.05, max_lag=12):
    """Run PC algorithm for causal discovery with time lags"""
    return run_pc_algorithm_efficient(df, alpha, max_lag)


def parse_lagged_var_name(var_name):
    """Parse lagged variable name like 'Y1_t-2' to ('Y1', 2)"""
    if '_t-' in var_name:
        name, lag_str = var_name.split('_t-')
        return name, int(lag_str)
    elif '_t0' in var_name:
        name = var_name.replace('_t0', '')
        return name, 0
    else:
        return var_name, 0


def run_pc_analysis(df, target_column, confounders=None, max_lag=12, alpha=0.05, dataset_name=None):
    """Run PC algorithm causal discovery analysis."""
    
    print("Running PC algorithm analysis...")
    
    if confounders is None:
        confounders = []
    
    # Remove confounders from analysis
    analysis_cols = [col for col in df.columns if col not in confounders]
    df_analysis = df[analysis_cols].copy()
    
    try:
        edges, adj_matrix = run_pc_algorithm(df_analysis, alpha, max_lag)
        # Add dataset name to each edge
        if edges and dataset_name:
            for edge in edges:
                edge['dataset'] = dataset_name
        return edges, adj_matrix, list(df_analysis.columns)
    except Exception as e:
        print(f"PC algorithm failed: {e}")
        return [], None, list(df_analysis.columns)


def extract_pc_edges(edges_list, threshold=0.1, dataset_name=None):
    """Extract significant causal edges from PC algorithm results."""
    
    if not edges_list:
        return pd.DataFrame()
    
    # Filter edges by significance and strength
    significant_edges = []
    for edge in edges_list:
        if edge['p_value'] < 0.05 and edge['abs_correlation'] > threshold:
            edge['dataset'] = dataset_name or 'Unknown'
            significant_edges.append(edge)
    
    return pd.DataFrame(significant_edges)


def calculate_transfer_entropy_simple(source, target, lag=1):
    """Calculate Transfer Entropy using a simpler, more robust approach."""
    
    try:
        n = len(source)
        if n <= lag + 1:
            return np.nan
        
        # Simple approach: correlation between lagged source and current target
        # controlling for lagged target
        source_lagged = source[:-lag]
        target_current = target[lag:]
        target_lagged = target[:-lag]
        
        if len(source_lagged) != len(target_current) or len(target_lagged) != len(target_current):
            return np.nan
            
        if len(target_current) < 10:
            return np.nan
        
        # Use partial correlation as proxy for transfer entropy
        from sklearn.linear_model import LinearRegression
        
        # Regress target_current on target_lagged
        reg_target = LinearRegression().fit(target_lagged.reshape(-1, 1), target_current)
        residual_target = target_current - reg_target.predict(target_lagged.reshape(-1, 1))
        
        # Regress source_lagged on target_lagged  
        reg_source = LinearRegression().fit(target_lagged.reshape(-1, 1), source_lagged)
        residual_source = source_lagged - reg_source.predict(target_lagged.reshape(-1, 1))
        
        # Correlation between residuals approximates conditional dependence
        from scipy.stats import pearsonr
        corr, p_val = pearsonr(residual_source, residual_target)
        
        # Convert correlation to transfer entropy-like measure
        te_value = abs(corr) if p_val < 0.05 else 0.0
        return te_value
        
    except Exception as e:
        return np.nan


def calculate_transfer_entropy_sklearn(source, target, lag=1, k=2):
    """Calculate Transfer Entropy using sklearn's mutual information as approximation."""
    
    if not SKLEARN_MI_AVAILABLE:
        return calculate_transfer_entropy_simple(source, target, lag)
    
    try:
        n = len(source)
        if n <= lag + k:
            return calculate_transfer_entropy_simple(source, target, lag)
        
        # Simpler approach - just use single lag for past
        target_future = target[lag:]
        target_past = target[:-lag].reshape(-1, 1)
        source_past = source[:-lag].reshape(-1, 1)
        
        # Ensure consistent lengths
        min_len = min(len(target_future), len(target_past), len(source_past))
        target_future = target_future[:min_len]
        target_past = target_past[:min_len]
        source_past = source_past[:min_len]
        
        if len(target_future) < 20:  # Need sufficient samples
            return calculate_transfer_entropy_simple(source, target, lag)
        
        # Combined features: target_past + source_past
        combined_past = np.hstack([target_past, source_past])
        
        # Calculate mutual information
        mi_full = mutual_info_regression(combined_past, target_future, random_state=42, discrete_features=False)[0]
        mi_target_only = mutual_info_regression(target_past, target_future, random_state=42, discrete_features=False)[0]
        
        te = mi_full - mi_target_only
        
        
        
        return max(0, te)
        
    except Exception as e:
        return calculate_transfer_entropy_simple(source, target, lag)


def run_transfer_entropy_analysis(df, target_column, confounders=None, max_lag=12, k=3, dataset_name=None):
    """Run Transfer Entropy analysis using sklearn approximation."""
    
    print("Running Transfer Entropy analysis...")
    
    if confounders is None:
        confounders = []
    
    # Remove confounders from analysis
    analysis_cols = [col for col in df.columns if col not in confounders]
    df_analysis = df[analysis_cols].copy()
    
    edges = []
    var_names = list(df_analysis.columns)
    
    for source_var in var_names:
        for target_var in var_names:
            if source_var == target_var:
                continue
                
            source_data = df_analysis[source_var].values
            target_data = df_analysis[target_var].values
            
            # Test different lags
            for lag in range(1, max_lag + 1):
                te_value = calculate_transfer_entropy_sklearn(source_data, target_data, lag, k)
                
                if not np.isnan(te_value) and te_value > 0.001:  # Lower threshold for sensitivity
                    edges.append({
                        'x1': source_var,
                        'x2': target_var,
                        'lag': lag,
                        'te_value': te_value,
                        'method': 'TransferEntropy',
                        'dataset': dataset_name or 'Unknown'
                    })
    
    return pd.DataFrame(edges)


def compare_all_methods(ccm_results_path, pc_edges, te_edges, output_folder, target_column, dataset_name=None):
    """Compare CCM/ECCM, PC algorithm, and Transfer Entropy results."""
    
    # Load CCM/ECCM results
    if ccm_results_path:
        try:
            ccm_results = pd.read_csv(ccm_results_path)
            # Check if CCM results are empty (just headers)
            if len(ccm_results) == 0:
                print(f"Warning: CCM results file is empty (placeholder) - using PC and TE only")
                ccm_results = pd.DataFrame()
            else:
                print(f"Loaded CCM results with {len(ccm_results)} edges")
        except:
            print(f"Could not load CCM results from {ccm_results_path}")
            ccm_results = pd.DataFrame()
    else:
        print("No CCM results provided - comparing PC and TE only")
        ccm_results = pd.DataFrame()
    
    # Get all unique edges from all methods
    all_edge_keys = set()
    
    # Collect edge keys from each method
    if not ccm_results.empty:
        ccm_edges = set(f"{row['x1']}_{row['x2']}" for _, row in ccm_results.iterrows())
        all_edge_keys.update(ccm_edges)
    
    if not pc_edges.empty:
        pc_edge_keys = set(f"{row['x1']}_{row['x2']}" for _, row in pc_edges.iterrows())
        all_edge_keys.update(pc_edge_keys)
    
    if not te_edges.empty:
        te_edge_keys = set(f"{row['x1']}_{row['x2']}" for _, row in te_edges.iterrows())
        all_edge_keys.update(te_edge_keys)
    
    if not all_edge_keys:
        print("No edges found in any method")
        return None, None, {}
    
    # Create comprehensive comparison table
    comparison_table = []
    
    for edge_key in sorted(all_edge_keys):
        x1, x2 = edge_key.split('_', 1)
        
        row = {
            'edge': edge_key,
            'x1': x1,
            'x2': x2,
            'dataset': dataset_name or 'Unknown'
        }
        
        # CCM/ECCM results
        ccm_row = ccm_results[(ccm_results['x1'] == x1) & (ccm_results['x2'] == x2)] if not ccm_results.empty else pd.DataFrame()
        if not ccm_row.empty:
            is_valid = ccm_row.iloc[0].get('is_Valid', 0)
            score = ccm_row.iloc[0].get('Score', np.nan)
            lag = ccm_row.iloc[0].get('timeToEffect', np.nan)
            row['CCM_ECCM_significant'] = 1 if is_valid == 2 else 0
            row['CCM_ECCM_score'] = score
            row['CCM_ECCM_lag'] = lag
        else:
            row['CCM_ECCM_significant'] = 0
            row['CCM_ECCM_score'] = np.nan
            row['CCM_ECCM_lag'] = np.nan
        
        # PC algorithm results 
        pc_row = pc_edges[(pc_edges['x1'] == x1) & (pc_edges['x2'] == x2)] if not pc_edges.empty else pd.DataFrame()
        if not pc_row.empty:
            # Take the best (highest significance) result if multiple lags
            best_pc = pc_row.loc[pc_row['p_value'].idxmin()]
            row['PC_significant'] = 1 if best_pc['p_value'] < 0.05 else 0
            row['PC_score'] = best_pc['abs_correlation']
            row['PC_lag'] = best_pc['lag']
            row['PC_p_value'] = best_pc['p_value']
        else:
            row['PC_significant'] = 0
            row['PC_score'] = np.nan
            row['PC_lag'] = np.nan
            row['PC_p_value'] = np.nan
        
        # Transfer Entropy results
        te_row = te_edges[(te_edges['x1'] == x1) & (te_edges['x2'] == x2)] if not te_edges.empty else pd.DataFrame()
        if not te_row.empty:
            # Take the best (highest TE value) result if multiple lags
            best_te = te_row.loc[te_row['te_value'].idxmax()]
            row['TE_significant'] = 1 if best_te['te_value'] > 0.01 else 0  # Threshold for TE significance
            row['TE_score'] = best_te['te_value'] 
            row['TE_lag'] = best_te['lag']
        else:
            row['TE_significant'] = 0
            row['TE_score'] = np.nan
            row['TE_lag'] = np.nan
        
        # Count how many methods detected this edge
        row['methods_detecting'] = row['CCM_ECCM_significant'] + row['PC_significant'] + row['TE_significant']
        
        comparison_table.append(row)
    
    df_comparison = pd.DataFrame(comparison_table)
    
    # Calculate summary statistics
    overlap_stats = {}
    
    # Method-specific counts
    ccm_significant = len(df_comparison[df_comparison['CCM_ECCM_significant'] == 1])
    pc_significant = len(df_comparison[df_comparison['PC_significant'] == 1])
    te_significant = len(df_comparison[df_comparison['TE_significant'] == 1])
    
    # Pairwise overlaps
    ccm_pc_both = len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 1) & (df_comparison['PC_significant'] == 1)])
    ccm_te_both = len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 1) & (df_comparison['TE_significant'] == 1)])
    pc_te_both = len(df_comparison[(df_comparison['PC_significant'] == 1) & (df_comparison['TE_significant'] == 1)])
    
    # All three methods
    all_three = len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 1) & 
                                 (df_comparison['PC_significant'] == 1) & 
                                 (df_comparison['TE_significant'] == 1)])
    
    overlap_stats = {
        'CCM_ECCM_total': ccm_significant,
        'PC_total': pc_significant, 
        'TE_total': te_significant,
        'CCM_PC_overlap': ccm_pc_both,
        'CCM_TE_overlap': ccm_te_both,
        'PC_TE_overlap': pc_te_both,
        'all_three_overlap': all_three,
        'CCM_PC_jaccard': ccm_pc_both / (ccm_significant + pc_significant - ccm_pc_both) if (ccm_significant + pc_significant - ccm_pc_both) > 0 else 0,
        'CCM_TE_jaccard': ccm_te_both / (ccm_significant + te_significant - ccm_te_both) if (ccm_significant + te_significant - ccm_te_both) > 0 else 0,
        'PC_TE_jaccard': pc_te_both / (pc_significant + te_significant - pc_te_both) if (pc_significant + te_significant - pc_te_both) > 0 else 0
    }
    
    # Save comprehensive comparison table
    df_comparison.to_csv(os.path.join(output_folder, 'comprehensive_methods_comparison.csv'), index=False)
    
    # Save only significant edges for cleaner view
    df_significant = df_comparison[df_comparison['methods_detecting'] > 0]
    df_significant.to_csv(os.path.join(output_folder, 'significant_edges_comparison.csv'), index=False)
    
    # Save statistics
    with open(os.path.join(output_folder, 'method_comparison_stats.txt'), 'w') as f:
        f.write("Method Comparison Statistics\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Significant Edges by Method:\n")
        f.write(f"CCM/ECCM: {ccm_significant} edges\n")
        f.write(f"PC Algorithm: {pc_significant} edges\n")
        f.write(f"Transfer Entropy: {te_significant} edges\n\n")
        
        f.write("Method Overlaps:\n")
        f.write(f"CCM & PC: {ccm_pc_both} edges (Jaccard: {overlap_stats['CCM_PC_jaccard']:.3f})\n")
        f.write(f"CCM & TE: {ccm_te_both} edges (Jaccard: {overlap_stats['CCM_TE_jaccard']:.3f})\n")
        f.write(f"PC & TE: {pc_te_both} edges (Jaccard: {overlap_stats['PC_TE_jaccard']:.3f})\n")
        f.write(f"All three methods: {all_three} edges\n")
        
        f.write(f"\nEdges detected by multiple methods: {len(df_significant[df_significant['methods_detecting'] > 1])}\n")
        f.write(f"Total unique edges: {len(df_comparison)}\n")
    
    return df_comparison, df_significant, overlap_stats


def create_publication_figures(df_comparison, df_significant, overlap_stats, output_folder):
    """Create publication-quality figures focused on meaningful comparisons."""
    
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    # Figure 1: Method comparison and overlap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Significant edge counts by method
    method_counts = {
        'CCM/ECCM': overlap_stats['CCM_ECCM_total'],
        'PC Algorithm': overlap_stats['PC_total'],
        'Transfer Entropy': overlap_stats['TE_total']
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(method_counts.keys(), method_counts.values(), color=colors)
    ax1.set_title('Significant Causal Links by Method')
    ax1.set_ylabel('Number of Significant Links')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Jaccard similarity heatmap
    methods = ['CCM/ECCM', 'PC', 'TE']
    jaccard_matrix = np.array([
        [1.0, overlap_stats['CCM_PC_jaccard'], overlap_stats['CCM_TE_jaccard']],
        [overlap_stats['CCM_PC_jaccard'], 1.0, overlap_stats['PC_TE_jaccard']],
        [overlap_stats['CCM_TE_jaccard'], overlap_stats['PC_TE_jaccard'], 1.0]
    ])
    
    im = ax2.imshow(jaccard_matrix, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(methods)
    ax2.set_yticklabels(methods)
    ax2.set_title('Method Agreement (Jaccard Index)')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax2.text(j, i, f'{jaccard_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_comparison_publication.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Method overlap visualization (Upset-style)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create overlap categories with better labels
    overlap_data = {
        'CCM/ECCM only': len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 1) & 
                                         (df_comparison['PC_significant'] == 0) & 
                                         (df_comparison['TE_significant'] == 0)]),
        'PC only': len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 0) & 
                                   (df_comparison['PC_significant'] == 1) & 
                                   (df_comparison['TE_significant'] == 0)]),
        'TE only': len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 0) & 
                                   (df_comparison['PC_significant'] == 0) & 
                                   (df_comparison['TE_significant'] == 1)]),
        'CCM/ECCM & PC': len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 1) & 
                                         (df_comparison['PC_significant'] == 1) & 
                                         (df_comparison['TE_significant'] == 0)]),
        'CCM/ECCM & TE': len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 1) & 
                                         (df_comparison['PC_significant'] == 0) & 
                                         (df_comparison['TE_significant'] == 1)]),
        'PC & TE': len(df_comparison[(df_comparison['CCM_ECCM_significant'] == 0) & 
                                   (df_comparison['PC_significant'] == 1) & 
                                   (df_comparison['TE_significant'] == 1)]),
        'All three methods': overlap_stats['all_three_overlap']
    }
    
    # Filter out zero counts for cleaner visualization
    overlap_data = {k: v for k, v in overlap_data.items() if v > 0}
    
    if overlap_data:
        bars = ax.barh(list(overlap_data.keys()), list(overlap_data.values()), 
                      color=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'mediumpurple', 'orange', 'red'][:len(overlap_data)])
        ax.set_title('Causal Link Detection Patterns Across Methods')
        ax.set_xlabel('Number of Links')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.annotate(f'{int(width)}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0),  # 3 points horizontal offset
                       textcoords="offset points",
                       ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_overlap_patterns.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Score comparison for edges detected by multiple methods
    if not df_significant.empty:
        multi_method_edges = df_significant[df_significant['methods_detecting'] > 1]
        
        if not multi_method_edges.empty:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # CCM vs PC scores
            ccm_pc_edges = multi_method_edges[(multi_method_edges['CCM_ECCM_significant'] == 1) & 
                                            (multi_method_edges['PC_significant'] == 1)]
            if not ccm_pc_edges.empty:
                ccm_scores = ccm_pc_edges['CCM_ECCM_score'].dropna()
                pc_scores = ccm_pc_edges['PC_score'].dropna()
                if len(ccm_scores) > 0 and len(pc_scores) > 0:
                    axes[0].scatter(ccm_scores, pc_scores, alpha=0.7, s=60)
                    axes[0].set_xlabel('CCM/ECCM Score')
                    axes[0].set_ylabel('PC Correlation')
                    axes[0].set_title('CCM/ECCM vs PC Algorithm')
                    axes[0].grid(True, alpha=0.3)
                    
                    try:
                        corr = np.corrcoef(ccm_scores, pc_scores)[0, 1]
                        axes[0].text(0.05, 0.95, f'r = {corr:.3f}', 
                                   transform=axes[0].transAxes,
                                   bbox=dict(boxstyle="round", facecolor='wheat'))
                    except:
                        pass
            
            # CCM vs TE scores
            ccm_te_edges = multi_method_edges[(multi_method_edges['CCM_ECCM_significant'] == 1) & 
                                            (multi_method_edges['TE_significant'] == 1)]
            if not ccm_te_edges.empty:
                ccm_scores = ccm_te_edges['CCM_ECCM_score'].dropna()
                te_scores = ccm_te_edges['TE_score'].dropna()
                if len(ccm_scores) > 0 and len(te_scores) > 0:
                    axes[1].scatter(ccm_scores, te_scores, alpha=0.7, s=60)
                    axes[1].set_xlabel('CCM/ECCM Score')
                    axes[1].set_ylabel('Transfer Entropy')
                    axes[1].set_title('CCM/ECCM vs Transfer Entropy')
                    axes[1].grid(True, alpha=0.3)
                    
                    try:
                        corr = np.corrcoef(ccm_scores, te_scores)[0, 1]
                        axes[1].text(0.05, 0.95, f'r = {corr:.3f}', 
                                   transform=axes[1].transAxes,
                                   bbox=dict(boxstyle="round", facecolor='wheat'))
                    except:
                        pass
            
            # PC vs TE scores
            pc_te_edges = multi_method_edges[(multi_method_edges['PC_significant'] == 1) & 
                                           (multi_method_edges['TE_significant'] == 1)]
            if not pc_te_edges.empty:
                pc_scores = pc_te_edges['PC_score'].dropna()
                te_scores = pc_te_edges['TE_score'].dropna()
                if len(pc_scores) > 0 and len(te_scores) > 0:
                    axes[2].scatter(pc_scores, te_scores, alpha=0.7, s=60)
                    axes[2].set_xlabel('PC Correlation')
                    axes[2].set_ylabel('Transfer Entropy')
                    axes[2].set_title('PC Algorithm vs Transfer Entropy')
                    axes[2].grid(True, alpha=0.3)
                    
                    try:
                        corr = np.corrcoef(pc_scores, te_scores)[0, 1]
                        axes[2].text(0.05, 0.95, f'r = {corr:.3f}', 
                                   transform=axes[2].transAxes,
                                   bbox=dict(boxstyle="round", facecolor='wheat'))
                    except:
                        pass
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'score_correlations.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Publication figures saved to {output_folder}")

def create_pc_te_figures(df_comparison, df_significant, overlap_stats, output_folder):
    """Create publication-quality figures for PC and TE only comparison."""
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set publication style
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    print(f"Creating PC-TE comparison figures in {output_folder}...")
    
    # Figure 1: Method comparison and agreement (PC vs TE only)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Method performance
    methods = ['PC Algorithm', 'Transfer Entropy']
    edge_counts = [overlap_stats['PC_total'], overlap_stats['TE_total']]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(methods, edge_counts, color=colors, alpha=0.8)
    ax1.set_ylabel('Number of Significant Causal Links')
    ax1.set_title('Causal Discovery Method Performance')
    ax1.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for bar, count in zip(bars, edge_counts):
        height = bar.get_height()
        ax1.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Method agreement heatmap (2x2 for PC vs TE)
    jaccard_matrix = np.array([[1.0, overlap_stats.get('PC_TE_jaccard', 0)],
                              [overlap_stats.get('PC_TE_jaccard', 0), 1.0]])
    
    im = ax2.imshow(jaccard_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['PC', 'TE'])
    ax2.set_yticklabels(['PC', 'TE'])
    ax2.set_title('Method Agreement (Jaccard Index)')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{jaccard_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_comparison_publication.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Method overlap visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create overlap categories for PC and TE only
    overlap_data = {
        'PC only': len(df_comparison[(df_comparison['PC_significant'] == 1) & 
                                   (df_comparison['TE_significant'] == 0)]),
        'TE only': len(df_comparison[(df_comparison['PC_significant'] == 0) & 
                                   (df_comparison['TE_significant'] == 1)]),
        'Both methods': overlap_stats.get('PC_TE_overlap', 0)
    }
    
    # Filter out zero counts for cleaner visualization
    overlap_data = {k: v for k, v in overlap_data.items() if v > 0}
    
    if overlap_data:
        bars = ax.barh(list(overlap_data.keys()), list(overlap_data.values()), 
                      color=['#ff7f0e', '#2ca02c', 'gold'][:len(overlap_data)])
        ax.set_title('Causal Link Detection Patterns Across Methods')
        ax.set_xlabel('Number of Links')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.annotate(f'{int(width)}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0),  # 3 points horizontal offset
                       textcoords="offset points",
                       ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_overlap_patterns.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Score comparison for edges detected by both methods
    if not df_significant.empty:
        both_methods_edges = df_significant[(df_significant['PC_significant'] == 1) & 
                                          (df_significant['TE_significant'] == 1)]
        
        if not both_methods_edges.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            pc_scores = both_methods_edges['PC_score'].dropna()
            te_scores = both_methods_edges['TE_score'].dropna()
            
            if len(pc_scores) > 0 and len(te_scores) > 0:
                ax.scatter(pc_scores, te_scores, alpha=0.7, s=60, color='purple')
                ax.set_xlabel('PC Correlation Score')
                ax.set_ylabel('Transfer Entropy Score')
                ax.set_title('Score Comparison for Edges Detected by Both Methods')
                ax.grid(True, alpha=0.3)
                
                try:
                    corr = np.corrcoef(pc_scores, te_scores)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                           transform=ax.transAxes,
                           bbox=dict(boxstyle="round", facecolor='wheat'))
                except:
                    pass
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'score_correlations.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print("PC-TE comparison figures created successfully!")

def get_ground_truth_edges(dataset_name):
    """Get ground truth causal edges for each synthetic dataset based on structural causal models."""
    
    ground_truth = {
        'dataset1_baseline': [
            ('Y1', 'Y1', 1),  # Y1(t) = 0.8 * Y1(t-1)
            ('Y2', 'Y2', 1),  # Y2(t) = 0.77 * Y2(t-1)
            ('Y1', 'Y3', 1),  # Y3(t) = 0.65 * Y1(t-1) - 0.79 * Y2(t-1)
            ('Y2', 'Y3', 1),  
            ('Y4', 'Y4', 1),  # Y4(t) = 0.65 * Y4(t-1)
            ('Y2', 'Y5', 1),  # Y5(t) = -0.3 * Y2(t-1) + 0.4 * Y3(t-1) + 0.55 * Y4(t-1)
            ('Y3', 'Y5', 1),
            ('Y4', 'Y5', 1)
        ],
        'dataset2_more_noise': [
            # Same structure as baseline, just with added noise
            ('Y1', 'Y1', 1),
            ('Y2', 'Y2', 1),
            ('Y1', 'Y3', 1),
            ('Y2', 'Y3', 1),
            ('Y4', 'Y4', 1),
            ('Y2', 'Y5', 1),
            ('Y3', 'Y5', 1),
            ('Y4', 'Y5', 1)
        ],
        'dataset3_add_x0': [
            ('Y0', 'Y0', 1),  # Y0(t) = 0.75 * Y0(t-1)
            ('Y1', 'Y1', 1),  # Y1(t) = 0.8 * Y1(t-1)
            ('Y2', 'Y2', 1),  # Y2(t) = 0.77 * Y2(t-1) + 0.4 * Y0(t-1)
            ('Y0', 'Y2', 1),
            ('Y1', 'Y3', 1),  # Y3(t) = 0.65 * Y1(t-1) - 0.79 * Y2(t-1)
            ('Y2', 'Y3', 1),
            ('Y4', 'Y4', 1),  # Y4(t) = 0.65 * Y4(t-1)
            ('Y2', 'Y5', 1),  # Y5(t) = -0.3 * Y2(t-1) + 0.4 * Y3(t-1) + 0.55 * Y4(t-1)
            ('Y3', 'Y5', 1),
            ('Y4', 'Y5', 1)
        ],
        'dataset4_add_x4a': [
            ('Y1', 'Y1', 1),  # Y1(t) = 0.8 * Y1(t-1)
            ('Y2', 'Y2', 1),  # Y2(t) = 0.77 * Y2(t-1)
            ('Y1', 'Y3', 1),  # Y3(t) = 0.65 * Y1(t-1) - 0.79 * Y2(t-1)
            ('Y2', 'Y3', 1),
            ('Y4', 'Y4', 1),  # Y4(t) = 0.65 * Y4(t-1)
            ('Y2', 'Y5', 1),  # Y5(t) = -0.3 * Y2(t-1) + 0.4 * Y3(t-1) + 0.55 * Y4(t-1) + 0.35 * Y4a(t-1)
            ('Y3', 'Y5', 1),
            ('Y4', 'Y5', 1),
            ('Y4a', 'Y5', 1),
            ('Y4a', 'Y4a', 1)  # Y4a(t) = 0.7 * Y4a(t-1)
        ]
    }
    
    if dataset_name not in ground_truth:
        print(f"Warning: No ground truth defined for {dataset_name}")
        return []
    
    # Convert to edge format consistent with our analysis
    edges = []
    for source, target, lag in ground_truth[dataset_name]:
        edge_name = f"{source}_{target}"
        edges.append(edge_name)
    
    return edges

def calculate_performance_metrics(predicted_edges, ground_truth_edges, dataset_name):
    """Calculate accuracy, precision, recall, F1, false positive rate."""
    
    predicted_set = set(predicted_edges) if predicted_edges else set()
    ground_truth_set = set(ground_truth_edges)
    
    # True positives: correctly identified causal edges
    tp = len(predicted_set & ground_truth_set)
    
    # False positives: incorrectly identified edges (predicted but not true)
    fp = len(predicted_set - ground_truth_set)
    
    # False negatives: missed edges (true but not predicted)
    fn = len(ground_truth_set - predicted_set)
    
    # True negatives: correctly identified non-causal relationships
    # For this, we need to consider all possible edges
    all_possible_edges = predicted_set | ground_truth_set
    tn = len(all_possible_edges) - tp - fp - fn if len(all_possible_edges) > 0 else 0
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'dataset': dataset_name,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': fpr,
        'ground_truth_edges': len(ground_truth_set),
        'predicted_edges': len(predicted_set)
    }
    
    return metrics

def add_performance_metrics_to_comparison(df_comparison, dataset_name, output_folder):
    """Add performance metrics for each method to comparison results."""
    
    # Get ground truth
    ground_truth_edges = get_ground_truth_edges(dataset_name)
    
    if not ground_truth_edges:
        print(f"No ground truth available for {dataset_name} - skipping performance metrics")
        return None
    
    # Extract predicted edges for each method
    ccm_edges = df_comparison[df_comparison['CCM_ECCM_significant'] == 1]['edge'].tolist()
    pc_edges = df_comparison[df_comparison['PC_significant'] == 1]['edge'].tolist()
    te_edges = df_comparison[df_comparison['TE_significant'] == 1]['edge'].tolist()
    
    # Calculate metrics for each method
    methods_metrics = {}
    
    if ccm_edges:  # CCM results available
        methods_metrics['CCM_ECCM'] = calculate_performance_metrics(ccm_edges, ground_truth_edges, dataset_name)
    
    if pc_edges:
        methods_metrics['PC'] = calculate_performance_metrics(pc_edges, ground_truth_edges, dataset_name)
        
    if te_edges:
        methods_metrics['TE'] = calculate_performance_metrics(te_edges, ground_truth_edges, dataset_name)
    
    # Create summary dataframe
    metrics_summary = []
    for method, metrics in methods_metrics.items():
        metrics_summary.append({
            'Method': method,
            'Dataset': dataset_name,
            'True_Positives': metrics['true_positives'],
            'False_Positives': metrics['false_positives'],
            'False_Negatives': metrics['false_negatives'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'False_Positive_Rate': metrics['false_positive_rate'],
            'Ground_Truth_Edges': metrics['ground_truth_edges'],
            'Predicted_Edges': metrics['predicted_edges']
        })
    
    metrics_df = pd.DataFrame(metrics_summary)
    
    # Save performance metrics
    metrics_file = os.path.join(output_folder, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    # Create detailed report
    report_file = os.path.join(output_folder, 'performance_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"Performance Metrics Report for {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Ground Truth Edges ({len(ground_truth_edges)}): {', '.join(ground_truth_edges)}\n\n")
        
        for method, metrics in methods_metrics.items():
            f.write(f"{method} Method:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Predicted Edges ({metrics['predicted_edges']}): ")
            if method == 'CCM_ECCM':
                predicted = ccm_edges
            elif method == 'PC':
                predicted = pc_edges
            else:
                predicted = te_edges
            f.write(f"{', '.join(predicted) if predicted else 'None'}\n")
            f.write(f"  True Positives: {metrics['true_positives']}\n")
            f.write(f"  False Positives: {metrics['false_positives']}\n")
            f.write(f"  False Negatives: {metrics['false_negatives']}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
            f.write(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}\n\n")
    
    print(f"Performance metrics saved to {metrics_file}")
    print(f"Detailed report saved to {report_file}")
    
    return methods_metrics

def main():
    parser = argparse.ArgumentParser(description="Compare CCM/ECCM with PCMCI and Transfer Entropy")
    parser.add_argument('--file_path', type=str, required=True, 
                       help='Path to the input CSV file')
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='Path to output folder')
    parser.add_argument('--target_column', type=str, required=True, 
                       help='Target column for analysis')
    parser.add_argument('--confounders', type=str, default="", 
                       help='Comma-separated list of confounder columns')
    parser.add_argument('--z_score_threshold', type=float, default=3.0, 
                       help='Z-score threshold for outlier detection')
    parser.add_argument('--resample_freq', type=str, default='D', 
                       help='Frequency for data resampling')
    parser.add_argument('--max_lag', type=int, default=12, 
                       help='Maximum lag for analysis')
    parser.add_argument('--alpha', type=float, default=0.05, 
                       help='Significance level for PCMCI')
    parser.add_argument('--pc_threshold', type=float, default=0.1,
                       help='Minimum correlation threshold for PC algorithm edges')
    parser.add_argument('--te_k', type=int, default=3,
                       help='History length for Transfer Entropy')
    parser.add_argument('--ccm_results_file', type=str, default='',
                       help='Path to existing CCM results file')

    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(args.file_path)
    df_processed, scalers = preprocess_data(df, args.z_score_threshold, args.resample_freq)
    
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Variables: {list(df_processed.columns)}")
    
    # Parse confounders
    confounders = [c.strip() for c in args.confounders.split(',') if c.strip()]
    
    # Extract dataset name from file path or use default
    dataset_name = os.path.basename(args.file_path).replace('.csv', '') if args.file_path else 'Unknown'
    
    # Run PC algorithm analysis
    pc_edges = pd.DataFrame()
    print("\nRunning PC algorithm analysis...")
    try:
        edges_list, adj_matrix, var_names = run_pc_analysis(
            df_processed, args.target_column, confounders, 
            args.max_lag, args.alpha, dataset_name
        )
        
        if edges_list:
            pc_edges = extract_pc_edges(edges_list, args.pc_threshold, dataset_name)
            print(f"PC algorithm found {len(pc_edges)} significant edges")
        else:
            print("PC algorithm found no significant edges")
    except Exception as e:
        print(f"PC algorithm analysis failed: {e}")
    
    # Run Transfer Entropy analysis
    print("\nRunning Transfer Entropy analysis...")
    te_edges = run_transfer_entropy_analysis(
        df_processed, args.target_column, confounders, args.max_lag, args.te_k, dataset_name
    )
    print(f"Transfer Entropy found {len(te_edges)} significant edges")
    
    # Save individual results with dataset identification
    
    if not pc_edges.empty:
        pc_edges.to_csv(os.path.join(args.output_folder, 'pc_results.csv'), index=False)
    
    if not te_edges.empty:
        te_edges.to_csv(os.path.join(args.output_folder, 'transfer_entropy_results.csv'), index=False)
    
    # Find CCM results file
    ccm_results_path = args.ccm_results_file
    if not ccm_results_path:
        # Try common CCM result file names
        possible_files = [
            'CCM_ECCM_curated.csv',
            'Surr_filtered.csv', 
            'CCM2_results.csv'
        ]
        for filename in possible_files:
            test_path = os.path.join(args.output_folder, filename)
            if os.path.exists(test_path):
                ccm_results_path = test_path
                break
    
    # Compare all methods
    if ccm_results_path and os.path.exists(ccm_results_path):
        print(f"\nComparing all methods with CCM results from: {ccm_results_path}")
        
        # Extract dataset name from file path or use default
        dataset_name = os.path.basename(args.file_path).replace('.csv', '') if args.file_path else 'Unknown'
        
        df_comparison, df_significant, overlap_stats = compare_all_methods(
            ccm_results_path, pc_edges, te_edges, args.output_folder, args.target_column, dataset_name
        )
        
        # Create publication figures
        print("Creating publication-quality figures...")
        create_publication_figures(df_comparison, df_significant, overlap_stats, args.output_folder)
        
        print("\nComparison completed!")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"CCM/ECCM: {overlap_stats['CCM_ECCM_total']} significant edges")
        print(f"PC Algorithm: {overlap_stats['PC_total']} significant edges")
        print(f"Transfer Entropy: {overlap_stats['TE_total']} significant edges")
        
        print("\nMethod Agreement (Jaccard Index):")
        print(f"CCM/ECCM vs PC: {overlap_stats['CCM_PC_jaccard']:.3f}")
        print(f"CCM/ECCM vs TE: {overlap_stats['CCM_TE_jaccard']:.3f}")
        print(f"PC vs TE: {overlap_stats['PC_TE_jaccard']:.3f}")
        print(f"All three methods: {overlap_stats['all_three_overlap']} edges")
        
    else:
        print("No CCM results found for comparison")
        print("Individual method results saved")
    
    print(f"\nAll results saved to: {args.output_folder}")
    print("Files created:")
    print("- pc_results.csv: PC algorithm edge results")
    print("- transfer_entropy_results.csv: Transfer Entropy results")
    if ccm_results_path and os.path.exists(ccm_results_path):
        print("- all_methods_comparison.csv: Detailed comparison")
        print("- method_comparison_stats.txt: Summary statistics")
        print("- Various visualization PNG files")


if __name__ == "__main__":
    main()
