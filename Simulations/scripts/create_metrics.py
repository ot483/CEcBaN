#!/usr/bin/env python3
import pandas as pd
import numpy as np

def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0
    balanced_acc = (recall + specificity) / 2
    f05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0
    f2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall) if (precision + recall) > 0 else 0
    tp_fp_ratio = tp / fp if fp > 0 else float('inf')
    return {
        'Precision': precision, 'Recall': recall, 'F1_Score': f1,
        'F0.5_Score': f05, 'F2_Score': f2, 'Accuracy': accuracy,
        'Balanced_Accuracy': balanced_acc, 'Specificity': specificity,
        'MCC': mcc, 'TP_FP_Ratio': tp_fp_ratio
    }

def calculate_edge_count_accuracy(predicted, ground_truth):
    error = abs(predicted - ground_truth)
    relative_error = error / ground_truth if ground_truth > 0 else 0
    return 1 - relative_error

df_granger_te = pd.read_csv('results/performance_comparison_with_ccm.csv')
df_pc = pd.read_csv('results/all_datasets_performance_metrics_CORRECTED.csv')

all_metrics = []

for method in ['CCM-ECCM', 'Granger', 'TE']:
    method_data = df_granger_te[df_granger_te['Method'] == method]
    for _, row in method_data.iterrows():
        metrics = calculate_metrics(row['TP'], row['FP'], row['FN'], row['TN'])
        edge_acc = calculate_edge_count_accuracy(row['Predicted'], row['Ground_Truth'])
        metrics['Edge_Count_Accuracy'] = edge_acc
        metrics.update({
            'Method': method, 'Dataset': row['Dataset'],
            'TP': row['TP'], 'FP': row['FP'], 'FN': row['FN'],
            'Predicted': row['Predicted'], 'Ground_Truth': row['Ground_Truth']
        })
        all_metrics.append(metrics)

method_data = df_pc[df_pc['Method'] == 'PC']
for _, row in method_data.iterrows():
    metrics = calculate_metrics(
        row['True_Positives'], row['False_Positives'],
        row['False_Negatives'], row['True_Negatives']
    )
    edge_acc = calculate_edge_count_accuracy(
        row['Predicted_Edges'], row['Ground_Truth_Edges']
    )
    metrics['Edge_Count_Accuracy'] = edge_acc
    metrics.update({
        'Method': 'PC', 'Dataset': row['Dataset'],
        'TP': row['True_Positives'], 'FP': row['False_Positives'],
        'FN': row['False_Negatives'],
        'Predicted': row['Predicted_Edges'],
        'Ground_Truth': row['Ground_Truth_Edges']
    })
    all_metrics.append(metrics)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('results/metrics.csv', index=False)

summary_data = []
for method in ['CCM-ECCM', 'PC', 'Granger', 'TE']:
    method_data = metrics_df[metrics_df['Method'] == method]
    if len(method_data) == 0:
        continue
    summary = {
        'Method': method,
        'Precision': method_data['Precision'].mean(),
        'Recall': method_data['Recall'].mean(),
        'Specificity': method_data['Specificity'].mean(),
        'F1_Score': method_data['F1_Score'].mean(),
        'F0.5_Score': method_data['F0.5_Score'].mean(),
        'MCC': method_data['MCC'].mean(),
        'Balanced_Acc': method_data['Balanced_Accuracy'].mean(),
        'Edge_Count_Acc': method_data['Edge_Count_Accuracy'].mean(),
        'TP_FP_Ratio': method_data['TP_FP_Ratio'].mean(),
        'Avg_Predicted': method_data['Predicted'].mean(),
        'Avg_GT': method_data['Ground_Truth'].mean(),
        'Total_TP': method_data['TP'].sum(),
        'Total_FP': method_data['FP'].sum()
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/summary.csv', index=False)
