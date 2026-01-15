#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv('results/summary.csv')
methods = df['Method'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel D
ax = axes[0]
predicted = [df[df['Method']==m]['Avg_Predicted'].values[0] for m in methods]
gt = df['Avg_GT'].values[0]
x = np.arange(len(methods))
bars = ax.bar(x, predicted, color='#5B9BD5', alpha=0.85, width=0.6)
ax.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([0, max(predicted) * 1.25])
ax.set_title('A) Edge Count Accuracy', fontsize=13, fontweight='bold', pad=10)

# Panel F
ax = axes[1]
spec = [df[df['Method']==m]['Specificity'].values[0] for m in methods]
x = np.arange(len(methods))
bars = ax.bar(x, spec, color='#5B9BD5', alpha=0.85, width=0.6)
ax.set_ylabel('Specificity (True Negative Rate)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim([0, 1.05])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_title('B) False Positive Control', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('results/panels_DF.png', dpi=300, bbox_inches='tight')
plt.savefig('results/panels_DF.pdf', dpi=300, bbox_inches='tight')
plt.close()
