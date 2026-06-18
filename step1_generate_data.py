"""
BANK CHURN PREDICTION PROJECT
Zimbabwe Banking/MicroFinance — Industrial Attachment

Step 1: Generate Dataset
========================
This script creates a realistic synthetic dataset that matches the
exact structure of the Kaggle 'Churn for Bank Customers' dataset.

HOW TO USE THE REAL KAGGLE DATASET INSTEAD:
  1. Go to: https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
  2. Sign in and click Download
  3. Extract and rename the file to: Churn_Modelling.csv
  4. Place it inside your  data/  folder
  5. In step2_eda.py, change DATA_PATH to point to that file

For now, this script generates a synthetic version so you can
run the full pipeline immediately without a Kaggle account.
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, 'Churn_Modelling.csv')

# ── Generate ──────────────────────────────────────────────────────────────────
print("=" * 55)
print(" STEP 1 — GENERATING DATASET")
print("=" * 55)

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'RowNumber':    range(1, n + 1),
    'CustomerId':   np.random.randint(15000000, 16000000, n),
    'Surname':      ['Customer_' + str(i) for i in range(n)],
    'CreditScore':  np.random.randint(350, 850, n),
    'Geography':    np.random.choice(
                        ['France', 'Germany', 'Spain'], n,
                        p=[0.50, 0.25, 0.25]),
    'Gender':       np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
    'Age':          np.random.randint(18, 75, n),
    'Tenure':       np.random.randint(0, 10, n),
    'Balance':      np.where(
                        np.random.rand(n) < 0.30,
                        0,
                        np.random.uniform(10000, 250000, n)),
    'NumOfProducts': np.random.choice([1, 2, 3, 4], n, p=[0.45, 0.46, 0.06, 0.03]),
    'HasCrCard':     np.random.choice([0, 1], n, p=[0.29, 0.71]),
    'IsActiveMember': np.random.choice([0, 1], n, p=[0.49, 0.51]),
    'EstimatedSalary': np.random.uniform(11.58, 199992.48, n),
})

# Churn label — correlated with real banking signals
churn_prob = (
    0.05
    + 0.15 * (df['Age'] > 45).astype(float)
    + 0.10 * (df['IsActiveMember'] == 0).astype(float)
    + 0.10 * (df['NumOfProducts'] > 2).astype(float)
    + 0.05 * (df['Balance'] == 0).astype(float)
    - 0.03 * (df['Tenure'] > 5).astype(float)
).clip(0.01, 0.99)

df['Exited'] = (np.random.rand(n) < churn_prob).astype(int)

df.to_csv(OUTPUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Rows generated : {len(df):,}")
print(f"  Columns        : {len(df.columns)}")
print(f"  Churn rate     : {df['Exited'].mean() * 100:.1f}%"
      f"  ({df['Exited'].sum():,} churned / {(df['Exited']==0).sum():,} retained)")
print(f"\n  Saved to: data/Churn_Modelling.csv")

print("\n  Column overview:")
for col in df.columns:
    print(f"    {col:<20} dtype={str(df[col].dtype):<10} "
          f"nulls={df[col].isnull().sum()}")

print("\n  Sample rows:")
print(df.head(3).to_string(index=False))

print("\n" + "=" * 55)
print(" DONE. Run step2_eda.py next.")
print("=" * 55)