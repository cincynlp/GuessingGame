import pandas as pd
from sklearn.metrics import cohen_kappa_score

def load_and_merge(file1, file2):
    """
    Load two annotator CSVs and merge them on identifying columns.
    Returns a DataFrame with suffixes _ann1 and _ann2 for labels.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    # Debug: preview loaded data
    print("Annotator1 sample rows:")
    print(df1.head(), "\n")
    print("Annotator2 sample rows:")
    print(df2.head(), "\n")
    # Normalize merge keys: strip whitespace, lowercase
    for col in ['object', 'mode', 'question']:
        df1[col] = df1[col].astype(str).str.strip().str.lower()
        df2[col] = df2[col].astype(str).str.strip().str.lower()
    # Ensure turn_idx is treated consistently
    df1['turn_idx'] = df1['turn_idx'].astype(int)
    df2['turn_idx'] = df2['turn_idx'].astype(int)
    # Build composite key for diagnostics
    df1['key'] = df1['object'] + '|' + df1['mode'] + '|' + df1['turn_idx'].astype(str) + '|' + df1['question']
    df2['key'] = df2['object'] + '|' + df2['mode'] + '|' + df2['turn_idx'].astype(str) + '|' + df2['question']
    print(f"Unique keys in annotator1: {df1['key'].nunique()}")
    print(f"Unique keys in annotator2: {df2['key'].nunique()}")
    keys = ['object', 'mode', 'turn_idx', 'question']
    merged = pd.merge(df1, df2, on=keys, suffixes=('_ann1', '_ann2'))
    print(f"Number of matched rows after merge: {len(merged)}")
    return merged

def compute_agreement(df):
    """
    Compute percent agreement and Cohen's kappa on the merged labels.
    """
    labels1 = df['label_ann1']
    labels2 = df['label_ann2']
    percent_agreement = (labels1 == labels2).mean()
    kappa = cohen_kappa_score(labels1, labels2)
    return percent_agreement, kappa





df = load_and_merge("Checker_Validation/checker_annotation_type.txt", "Checker_Validation/checker_annotation_typeDylan.txt")
if df.empty:
    print("No matching records found between the two CSVs.")

percent, kappa = compute_agreement(df)
print(f'Number of matched samples: {len(df)}')
print(f'Percent agreement: {percent:.2%}')
print(f"Cohen's kappa: {kappa:.3f}")

