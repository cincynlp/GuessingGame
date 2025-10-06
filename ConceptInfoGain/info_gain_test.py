#!/usr/bin/env python3
import os
import pandas as pd
import math
from lifelines import CoxPHFitter
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

bayes = False

# for file in os.listdir('ConceptInfoGain/threshold_experiments'):
# 2) Load IG CSV
ig_df = pd.read_csv("ConceptInfoGain/annotated/annotated_0.6_output.csv")
bayes_df = pd.read_csv('BayesInfoGain/results/final.txt')

# 3) Use IG DataFrame directly; derive question_idx and event by grouping
df = ig_df.copy()
df['question_idx'] = df.groupby('secret').cumcount() + 1
# mark the final question per secret as the event
df['total_qs'] = df.groupby('secret')['question_idx'].transform('max')
df['event'] = (df['question_idx'] == df['total_qs']).astype(int)

# 4) Compute questions remaining after each Q
df['remaining']  = df['total_qs'] - df['question_idx']

if bayes:
    bayes_df['remaining'] = df['remaining']
    bayes_df['event'] = df['event']
    bayes_df['question_idx'] = df['question_idx']
    df = bayes_df

# Drop any rows where IG is missing
df = df.dropna(subset=['ig'])
# Standardize IG to improve interpretability of Cox model
ig_mean = df['ig'].mean()
ig_std = df['ig'].std()
df['ig_z'] = (df['ig'] - ig_mean) / ig_std

# # Drop all rows where the label is "Direct"
# df = df[df['label'] != 'Direct']

# 5) Cox proportional hazards model
cph = CoxPHFitter()
# We need a DataFrame with duration_col, event_col, and covariates
cox_df = df[['question_idx','event','ig_z']].copy()
# cox_df.rename(columns={'ig_z': 'ig'}, inplace=True)
cph.fit(cox_df, duration_col='question_idx', event_col='event')
print("\n=== Cox PH Model ===")
print(cph.summary)

# 6) OLS Regression: Use raw IG to predict number of questions remaining
# print("\n=== OLS Regression ===")
# ig_df['question_idx'] = df['question_idx']
# ig_df['total_qs'] = df['total_qs']
# ig_df['remaining'] = ig_df['total_qs'] - ig_df['question_idx']
# ols_model = smf.ols('remaining ~ ig', data=ig_df).fit()
# print(ols_model.summary())

# 7) Accelerated Failure Time (AFT) model: Log-normal AFT model
print("\n=== AFT Model (Log-normal) ===")
from lifelines import LogNormalAFTFitter
aft = LogNormalAFTFitter()
aft_df = df[['question_idx', 'event', 'ig_z']].copy()
aft.fit(aft_df, duration_col='question_idx', event_col='event')
print(aft.summary)

# Compute per-game average IG and total questions
per_game = df.groupby('secret').agg(
    avg_ig=('ig', 'mean'),
    total_questions=('question_idx', 'max')
).reset_index()
# Spearman between avg_ig and total_questions
rho_game, p_game = spearmanr(per_game['avg_ig'], per_game['total_questions'])
print(f"\nSpearman correlation (avg IG vs. total questions): rho={rho_game:.3f}, p={p_game:.3e}")

# 9) Scatter plot
plt.figure(figsize=(6,4))
plt.scatter(df['ig'], df['remaining'], alpha=0.5)
plt.xlabel('Information Gain')
plt.ylabel('Questions Remaining')
if bayes:
    plt.title('Bayes IG vs. Questions Remaining')
else:
    plt.title('Entropy IG vs. Questions Remaining')
plt.grid(True)
plt.tight_layout()
plt.show()

# 10) Kaplan-Meier Survival Curve by IG (high vs. low)
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
df['ig_quintile'] = pd.qcut(df['ig'], q=6, labels=[f'Q{i+1}' for i in range(6)])

plt.figure(figsize=(6, 4))
for label, group in df.groupby('ig_quintile'):
    kmf.fit(group['question_idx'], event_observed=group['event'], label=f'IG {label}')
    kmf.plot_survival_function(ci_show=False)

if bayes:
    plt.title('Bayes- Kaplan-Meier Curve by Information Gain Quartile')
else:
    plt.title('Entropy- Kaplan-Meier Curve by Information Gain Quartile')
plt.xlabel('Question Index')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.tight_layout()
plt.show()