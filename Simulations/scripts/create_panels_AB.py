#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/summary.csv')
methods = ['CCM-ECCM', 'PC', 'Granger', 'TE']
df = df[df['Method'].isin(methods)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A
ax = axes[0]
edge_acc = df.set_index('Method')['Edge_Count_Acc'].reindex(methods)
colors = ['#66c2a5' if m == 'CCM-ECCM' else '#d3d3d3' for m in methods]
bars = ax.bar(range(len(methods)), edge_acc, color=colors, edgecolor='black', linewidth=2.5, alpha=0.9)
ax.set_ylabel('Edge Count Accuracy', fontweight='bold', fontsize=14)
ax.set_title('(A) Network Size Accuracy', fontweight='bold', fontsize=15, pad=15)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([-0.5, 0.8])
for i, (bar, val) in enumerate(zip(bars, edge_acc)):
    y_pos = val + 0.03 if val > 0 else 0.05
    ax.text(i, y_pos, f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)

# Panel B
ax = axes[1]
specificity = df.set_index('Method')['Specificity'].reindex(methods)
bars = ax.bar(range(len(methods)), specificity, color=colors, edgecolor='black', linewidth=2.5, alpha=0.9)
ax.set_ylabel('Specificity (True Negative Rate)', fontweight='bold', fontsize=14)
ax.set_title('(B) False Positive Control', fontweight='bold', fontsize=15, pad=15)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0.3, 0.9])
for i, (bar, val) in enumerate(zip(bars, specificity)):
    ax.text(i, val + 0.01, f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('results/panels_AB.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/panels_AB.png', dpi=300, bbox_inches='tight')
plt.close()
