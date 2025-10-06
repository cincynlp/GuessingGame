import pandas as pd


ig_df = pd.read_csv('ConceptInfoGain/threshold_experiments/annotated/annotated_0.6_output.csv')  # must have secret,question,ig

# Add code to compute various metrics

# 1. Proportion of each question type
type_counts = ig_df['label'].value_counts(normalize=True).rename("proportion")

from scipy.stats import bootstrap
import numpy as np

def compute_ci(series):
    if len(series) < 2:
        return (np.nan, np.nan)  # Avoid crashing on small samples
    res = bootstrap((series.values,), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    return res.confidence_interval.low, res.confidence_interval.high

avg_ig_with_ci = []
for label, group in ig_df.groupby('label'):
    mean_ig = group['ig'].mean()
    ci_low, ci_high = compute_ci(group['ig'])
    avg_ig_with_ci.append({'label': label, 'mean': mean_ig, 'ci_low': ci_low, 'ci_high': ci_high})

avg_ig_by_type = pd.DataFrame(avg_ig_with_ci).set_index('label')

# 3. Count of questions per type
count_by_type = ig_df['label'].value_counts().rename("count")

# 4. IG distribution per type per turn (if turn info is available)
# This requires knowing the turn number; if not available, skip this for now
ig_df['question_idx'] = ig_df.groupby('secret').cumcount() + 1

# 4. IG distribution per type per turn
ig_by_type_and_turn = ig_df.groupby(['label', 'question_idx'])['ig'].agg(['mean', 'count']).reset_index()

# Print a sample of the distribution table
print("\nMean IG per type per turn (sample):")
print(ig_by_type_and_turn.head(15))


# 8. Print summary
print("Proportion of each question type:")
print(type_counts)
print("\nAverage IG by question type with 95% CI:")
print(avg_ig_by_type.to_string(float_format="%.5f"))

# --- EMNLP-Style Plotting Block ---
import matplotlib.pyplot as plt
import seaborn as sns

# EMNLP-style plotting parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 1.2,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

# Compute percentage distribution of question types per turn
counts_by_turn = ig_df.groupby(['question_idx', 'label']).size().unstack(fill_value=0)
proportions_by_turn = counts_by_turn.div(counts_by_turn.sum(axis=1), axis=0)

# Reorder columns for consistency
question_type_order = ['Attribute', 'Function', 'Location', 'Category', 'Direct']
proportions_by_turn = proportions_by_turn[[col for col in question_type_order if col in proportions_by_turn.columns]]

# Plot: Question type proportions over time using stackplot for better labeling
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
x = proportions_by_turn.index
y = [proportions_by_turn[col].values for col in proportions_by_turn.columns]
labels = proportions_by_turn.columns

ax.stackplot(x, *y, labels=labels, alpha=0.95)
ax.set_title('Distribution of Question Types Over Time')
ax.set_xlabel('Turn Index')
ax.set_ylabel('Proportion')
ax.set_ylim(0, 1)
ax.set_xlim(left=1)
# ax.grid(True, linestyle='--', linewidth=0.1, alpha=0.7)  # grid removed for minimalism
ax.set_xticks(range(1, int(x.max()) + 1, 5))
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(width=1.2)
ax.legend(title='Question Type', loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
plt.savefig("figures/question_type_distribution.pdf", bbox_inches='tight', dpi=300)
plt.show()


ig_closed_vs_open_df = pd.read_csv('ConceptInfoGain/threshold_experiments/annotated/annotated_aspect_0.6_output.csv') 

# Compute percentage distribution of open vs. closed questions per turn
ig_closed_vs_open_df['question_idx'] = ig_closed_vs_open_df.groupby('secret').cumcount() + 1
counts_by_turn_aspect = ig_closed_vs_open_df.groupby(['question_idx', 'label']).size().unstack(fill_value=0)
proportions_by_turn_aspect = counts_by_turn_aspect.div(counts_by_turn_aspect.sum(axis=1), axis=0)

# Reorder columns for consistency
aspect_order = ['Open', 'Closed']
proportions_by_turn_aspect = proportions_by_turn_aspect[[col for col in aspect_order if col in proportions_by_turn_aspect.columns]]

# Use consistent styling
sns.set_palette("Set2")

# Plot: Open vs. Closed proportions over time using stackplot for better labeling
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
x = proportions_by_turn_aspect.index
y = [proportions_by_turn_aspect[col].values for col in proportions_by_turn_aspect.columns]
labels = proportions_by_turn_aspect.columns

ax.stackplot(x, *y, labels=labels, alpha=0.95)
ax.set_title('Distribution of Open vs. Closed Questions Over Time')
ax.set_xlabel('Turn Index')
ax.set_ylabel('Proportion')
ax.set_ylim(0, 1)
ax.set_xlim(left=1)
# ax.grid(True, linestyle='--', linewidth=0.1, alpha=0.7)  # grid removed for minimalism
ax.set_xticks(range(1, int(x.max()) + 1, 5))
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(width=1.2)
ax.legend(title='Aspect Type', loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
plt.savefig("figures/open_vs_closed_distribution.pdf", bbox_inches='tight', dpi=300)
plt.show()


# --- Statistical Comparison: Open vs. Closed IG ---
from scipy.stats import ttest_ind

# Compare IG between Open and Closed questions
open_ig = ig_closed_vs_open_df[ig_closed_vs_open_df['label'] == 'Open']['ig']
closed_ig = ig_closed_vs_open_df[ig_closed_vs_open_df['label'] == 'Closed']['ig']

# Compute means and 95% CIs for Open and Closed IG
def compute_ci(series):
    if len(series) < 2:
        return (np.nan, np.nan)
    res = bootstrap((series.values,), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    return res.confidence_interval.low, res.confidence_interval.high

open_mean = open_ig.mean()
open_ci_low, open_ci_high = compute_ci(open_ig)

closed_mean = closed_ig.mean()
closed_ci_low, closed_ci_high = compute_ci(closed_ig)

print(f"\nOpen Question IG: {open_mean:.5f} ± ({open_ci_low:.5f}, {open_ci_high:.5f})")
print(f"Closed Question IG: {closed_mean:.5f} ± ({closed_ci_low:.5f}, {closed_ci_high:.5f})")

t_stat, p_val = ttest_ind(open_ig, closed_ig, equal_var=False)
print(f"\nT-test for IG difference (Open vs. Closed):")
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4g}")