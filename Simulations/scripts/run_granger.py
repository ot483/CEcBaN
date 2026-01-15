#!/usr/bin/env python3
"""
Compare CCM-ECCM vs Granger Causality vs Transfer Entropy
All are proper causal inference methods
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def get_ground_truth_edges(dataset_name):
    """Define ground truth DIRECT edges"""
    if dataset_name in ['dataset1_baseline', 'dataset2_more_noise']:
        direct_edges = [
            ('Y1', 'Y3'), ('Y2', 'Y3'), ('Y2', 'Y5'), ('Y3', 'Y5'), ('Y4', 'Y5')
        ]
        variables = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']
    elif dataset_name == 'dataset3_add_x0':
        direct_edges = [
            ('Y0', 'Y2'), ('Y1', 'Y3'), ('Y2', 'Y3'), ('Y2', 'Y5'), ('Y3', 'Y5'), ('Y4', 'Y5')
        ]
        variables = ['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5']
    elif dataset_name == 'dataset4_add_x4a':
        direct_edges = [
            ('Y1', 'Y3'), ('Y2', 'Y3'), ('Y2', 'Y5'), ('Y3', 'Y5'), ('Y4', 'Y5'), ('Y4a', 'Y5')
        ]
        variables = ['Y1', 'Y2', 'Y3', 'Y4', 'Y4a', 'Y5']
    return direct_edges, variables

def get_all_true_paths(direct_edges, variables):
    """Calculate ALL true causal paths (direct + indirect)"""
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    G.add_edges_from(direct_edges)

    all_true_edges = set()
    for source in variables:
        for target in variables:
            if source != target and nx.has_path(G, source, target):
                all_true_edges.add((source, target))
    return all_true_edges

def load_method_results(dataset_name):
    """Load results from all three methods"""
    results = {}

    # Load CCM-ECCM
    ccm_file = f'quick_test_results/CCM_ECCM_curated_{dataset_name.replace("_", "")}.csv'
    try:
        ccm_df = pd.read_csv(ccm_file)
        ccm_edges = set()
        for _, row in ccm_df.iterrows():
            if 'is_Valid' in ccm_df.columns and row['is_Valid'] == 2:
                ccm_edges.add((row['x1'], row['x2']))
        results['CCM-ECCM'] = ccm_edges
        print(f"  CCM-ECCM: {len(ccm_edges)} edges")
    except:
        results['CCM-ECCM'] = set()
        print(f"  CCM-ECCM: Not found")

    # Load Granger Causality
    granger_file = f'proper_causal_discovery_results/{dataset_name}_granger.csv'
    try:
        granger_df = pd.read_csv(granger_file)
        granger_edges = set()
        for _, row in granger_df.iterrows():
            granger_edges.add((row['source'], row['target']))
        results['Granger'] = granger_edges
        print(f"  Granger: {len(granger_edges)} edges")
    except:
        results['Granger'] = set()
        print(f"  Granger: Not found")

    # Load Transfer Entropy (from comparison results)
    te_file = f'comprehensive_comparison_results/{dataset_name}_comparison/transfer_entropy_results.csv'
    try:
        te_df = pd.read_csv(te_file)
        te_edges = set()
        for _, row in te_df.iterrows():
            te_edges.add((row['x1'], row['x2']))
        results['TE'] = te_edges
        print(f"  Transfer Entropy: {len(te_edges)} edges")
    except:
        results['TE'] = set()
        print(f"  Transfer Entropy: Not found")

    return results

def calculate_performance(predicted, true_edges, all_possible_edges):
    """Calculate precision, recall, F1"""
    tp = len(predicted & true_edges)
    fp = len(predicted - true_edges)
    fn = len(true_edges - predicted)
    tn = all_possible_edges - tp - fp - fn

    precision = tp / len(predicted) if len(predicted) > 0 else 0
    recall = tp / len(true_edges) if len(true_edges) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / all_possible_edges if all_possible_edges > 0 else 0

    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Accuracy': accuracy,
        'Predicted': len(predicted), 'Ground_Truth': len(true_edges)
    }

def main():
    print("="*80)
    print("PROPER CAUSAL COMPARISON: CCM-ECCM vs GRANGER vs TRANSFER ENTROPY")
    print("="*80)

    datasets = [
        'dataset1_baseline',
        'dataset2_more_noise',
        'dataset3_add_x0',
        'dataset4_add_x4a'
    ]

    all_metrics = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Get ground truth
        direct_edges, variables = get_ground_truth_edges(dataset)
        true_paths = get_all_true_paths(direct_edges, variables)
        n_vars = len(variables)
        all_possible = n_vars * (n_vars - 1)

        print(f"\nGround Truth:")
        print(f"  Direct edges: {len(direct_edges)}")
        print(f"  Total causal paths: {len(true_paths)}")

        # Load results
        print(f"\nDetected edges:")
        results = load_method_results(dataset)

        # Calculate performance for each method
        print(f"\nPerformance:")
        for method_name, predicted_edges in results.items():
            if len(predicted_edges) > 0:
                perf = calculate_performance(predicted_edges, true_paths, all_possible)
                perf['Method'] = method_name
                perf['Dataset'] = dataset
                all_metrics.append(perf)

                print(f"\n  {method_name}:")
                print(f"    Precision: {perf['Precision']:.3f}")
                print(f"    Recall: {perf['Recall']:.3f}")
                print(f"    F1-Score: {perf['F1']:.3f}")
                print(f"    TP/FP: {perf['TP']}/{perf['FP']}")

    # Save results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('proper_causal_discovery_results/performance_comparison.csv', index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE (Average Across Datasets)")
    print(f"{'='*80}")

    for method in ['CCM-ECCM', 'Granger', 'TE']:
        method_data = metrics_df[metrics_df['Method'] == method]
        if len(method_data) > 0:
            print(f"\n{method}:")
            print(f"  Precision: {method_data['Precision'].mean():.3f}")
            print(f"  Recall: {method_data['Recall'].mean():.3f}")
            print(f"  F1-Score: {method_data['F1'].mean():.3f}")
            print(f"  Accuracy: {method_data['Accuracy'].mean():.3f}")
            print(f"  Avg Edges: {method_data['Predicted'].mean():.1f}")
            print(f"  Total TP: {method_data['TP'].sum()}")
            print(f"  Total FP: {method_data['FP'].sum()}")

    # Create comparison figure
    create_comparison_figure(metrics_df)

   

def create_comparison_figure(metrics_df):
    """Create clean comparison figure"""
    plt.style.use('default')
    plt.rcParams.update({'font.size': 13})

    fig, ax = plt.subplots(figsize=(12, 7))

    methods = ['CCM-ECCM', 'Granger', 'TE']
    method_labels = ['CCM-ECCM', 'Granger Causality', 'Transfer Entropy']
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    # Calculate averages
    avg_metrics = {}
    for method in methods:
        method_data = metrics_df[metrics_df['Method'] == method]
        if len(method_data) > 0:
            avg_metrics[method] = {
                'Precision': method_data['Precision'].mean(),
                'Recall': method_data['Recall'].mean(),
                'F1': method_data['F1'].mean(),
                'Accuracy': method_data['Accuracy'].mean()
            }

    x = np.arange(len(metrics))
    width = 0.25

    for i, method in enumerate(methods):
        if method in avg_metrics:
            values = [avg_metrics[method][m] for m in metrics]
            ax.bar(x + i*width, values, width, label=method_labels[i],
                   color=colors[i], alpha=0.85, edgecolor='black', linewidth=2)

            # Add value labels
            for j, v in enumerate(values):
                is_best = False
                if method in avg_metrics:
                    all_vals = [avg_metrics[m][metrics[j]] for m in methods if m in avg_metrics]
                    if abs(v - max(all_vals)) < 0.001:
                        is_best = True

                ax.text(j + i*width, v + 0.02, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=11,
                       fontweight='bold' if is_best else 'normal',
                       color='darkgreen' if is_best else 'black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                alpha=0.4, edgecolor='darkgreen', linewidth=1.5) if is_best else None)

    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=15)
    ax.set_xlabel('Performance Metric', fontweight='bold', fontsize=15)
    ax.set_title('Proper Causal Discovery Methods Comparison\n' +
                 'CCM-ECCM vs Granger Causality vs Transfer Entropy',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    note = "All three methods test for causal relationships (not just correlation)"
    fig.text(0.5, 0.01, note, ha='center', fontsize=10, style='italic', color='darkgreen', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('proper_causal_discovery_results/PROPER_CAUSAL_COMPARISON.png', dpi=300, bbox_inches='tight')
    plt.savefig('proper_causal_discovery_results/PROPER_CAUSAL_COMPARISON.pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Figure saved")

if __name__ == "__main__":
    main()
