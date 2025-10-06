from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import binom_test
from scipy.stats import sem, t
import pandas as pd
import os
import sys
import numpy as np


def compute_metrics(input_path):
    """
    Reads and parses the file, applies filter (n<50), returns
    (successes, total, n_values_array, average_n)
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 2)  # Only split on first two commas
            if len(parts) < 2:
                parts = line.strip().split('	', 2)
            if len(parts) >= 2:
                obj = parts[0]
                try:
                    n = int(parts[1])
                except ValueError:
                    n = None
                text = parts[2] if len(parts) > 2 else ''
                data.append((obj, n, text))
    df = pd.DataFrame(data, columns=['object', 'n', 'text'])
    mask = (df['n'] < 50)
    filtered = df[mask]
    total = len(df)
    successes = len(filtered)
    n_values_array = filtered['n'].dropna().to_numpy()
    average_n = filtered['n'].mean()
    return successes, total, n_values_array, average_n

def main(input_path):
    successes, total, n_values_array, average_n = compute_metrics(input_path)
    filtered_n = len(n_values_array)
    success_rate = successes / total if total > 0 else float('nan')
    ci_low, ci_upp = proportion_confint(successes, total, alpha=0.05, method='wilson')
    pval = binom_test(successes, total, prop=0.5)
    if filtered_n > 1:
        n_mean = average_n
        n_sem = sem(n_values_array)
        n_ci_low, n_ci_upp = t.interval(0.95, df=filtered_n-1, loc=n_mean, scale=n_sem)
    else:
        n_ci_low, n_ci_upp = float('nan'), float('nan')
    print()
    print(f"Results for: {input_path}")
    print(f"Success rate: {success_rate:.2%} (95% CI: {ci_low:.2%} - {ci_upp:.2%}, p={pval:.4f})")
    print(f"Average n: {average_n:.2f} (95% CI: {n_ci_low:.2f} - {n_ci_upp:.2f})")
    print()

def compare_results(path1, path2):
    succ1, tot1, n_vals1, avg1 = compute_metrics(path1)
    succ2, tot2, n_vals2, avg2 = compute_metrics(path2)
    # Two-proportion z-test for success rates
    count = np.array([succ1, succ2])
    nobs = np.array([tot1, tot2])
    stat, p_prop = proportions_ztest(count, nobs)
    # Independent t-test for average n
    stat_n, p_n = ttest_ind(n_vals1, n_vals2, nan_policy='omit')
    print()
    print(f"Comparison between: {path1} and {path2}")
    print(f"Success rates: {succ1}/{tot1} = {succ1/tot1:.2%}, {succ2}/{tot2} = {succ2/tot2:.2%}")
    print(f"Two-proportion z-test p-value: {p_prop}")
    print(f"Average n: {avg1:.2f} vs {avg2:.2f}")
    print(f"Independent t-test p-value (average n): {p_n:}")
    print()


# Chi-square test across multiple groups
def chi_square_test_across_groups(file_paths):
    """
    Perform a chi-square test to compare success rates across multiple conditions.
    Each file path corresponds to one condition.
    """
    successes = []
    totals = []
    for path in file_paths:
        succ, tot, _, _ = compute_metrics(path)
        successes.append(succ)
        totals.append(tot - succ)
    contingency_table = np.array([successes, totals])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("Chi-square test across groups:")
    print(f"Chi2 = {chi2}, p-value = {p}, degrees of freedom = {dof}")
    print()


# One-way ANOVA across multiple groups for average number of questions
def anova_test_on_n(file_paths):
    """
    Perform one-way ANOVA to compare average number of questions across conditions.
    Each file path corresponds to one condition.
    """
    n_arrays = []
    for path in file_paths:
        _, _, n_vals, _ = compute_metrics(path)
        n_arrays.append(n_vals)
    f_stat, p = f_oneway(*n_arrays)
    print("One-way ANOVA on average number of questions:")
    print(f"F = {f_stat}, p-value = {p}")
    print()

repeats_files = ["Raw Text Results Final/LLama3OpenNo3Repeats.txt", 
         "Raw Text Results Final/LLama3OpenNo3RepeatsYNResults.txt", 
         "Raw Text Results Final/LLama3OpenNoRepeatsOnlyOpenResults.txt", 
         "Raw Text Results Final/LLama3OpenNoRepeatsResults.txt",
         "Raw Text Results Final/LLama3OpenNoRepeatsYNResults.txt",
         ]

types_files = ["Raw Text Results Final/LLama3OpenAttributeResults.txt",
            "Raw Text Results Final/LLama3OpenFunctionResults.txt",
            "Raw Text Results Final/LLama3OpenLocationResults.txt",
            ]

model_files = ["Raw Text Results Final/GemOpen.txt",
            "Raw Text Results Final/GPTOpen.txt",
            ]

model_repeats = ["Raw Text Results Final/GemOpenNoRepeats.txt",
            "Raw Text Results Final/GPTOpenNoRepeats.txt",
            "Raw Text Results Final/LLama3OpenNoRepeatsOnlyOpenResults.txt"
            ]

# main("Raw Text Results Final/LLama3OpenResults.txt")
compare_results("Raw Text Results Final/LLamaNaturalOpenResults.txt", "Raw Text Results Final/LLama3YesResults.txt")