"""
BANK CHURN PREDICTION PROJECT
Zimbabwe Banking/MicroFinance — Industrial Attachment

Step 2: Exploratory Data Analysis & Feature Engineering
========================================================
Reads:  data/Churn_Modelling.csv
Writes: data/churn_engineered.csv
        outputs/eda_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Churn_Modelling.csv')
OUT_CSV     = os.path.join(BASE_DIR, 'data', 'churn_engineered.csv')
OUT_PLOT    = os.path.join(BASE_DIR, 'outputs', 'eda_analysis.png')
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — LOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(" STEP 2 — EDA & FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Churn rate: {df['Exited'].mean()*100:.1f}%  "
      f"({df['Exited'].sum():,} churned)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2 — BASIC EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Null values per column:")
print(df.isnull().sum().to_string())

print("\n[3] Churn by Geography:")
print(df.groupby('Geography')['Exited']
      .agg(['mean', 'count'])
      .rename(columns={'mean': 'churn_rate', 'count': 'customers'})
      .assign(churn_rate=lambda x: x['churn_rate'].map('{:.1%}'.format))
      .to_string())

print("\n[4] Churn by IsActiveMember:")
print(df.groupby('IsActiveMember')['Exited']
      .agg(['mean', 'count'])
      .rename(columns={'mean': 'churn_rate', 'count': 'customers'})
      .assign(churn_rate=lambda x: x['churn_rate'].map('{:.1%}'.format))
      .to_string())

print("\n[5] Numeric means — churned vs retained:")
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
comp = df.groupby('Exited')[num_cols].mean().T
comp.columns = ['Retained', 'Churned']
comp['Diff%'] = ((comp['Churned'] - comp['Retained'])
                 / comp['Retained'] * 100).round(1)
print(comp.round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — ZIMBABWE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" ZIMBABWE-SPECIFIC FEATURE ENGINEERING")
print("=" * 60)

df_zim = df.copy()

# 3.1 RFM Score
df_zim['Recency_Score']   = df_zim['IsActiveMember'].map({1: 1, 0: 3})
df_zim['Frequency_Score'] = df_zim['NumOfProducts'].clip(1, 4)
df_zim['Monetary_Score']  = pd.cut(
    df_zim['Balance'],
    bins=[-1, 0, 50000, 100000, 200000, np.inf],
    labels=[1, 2, 3, 4, 5]
).astype(int)
df_zim['RFM_Score'] = (
    df_zim['Recency_Score']   * 0.40 +
    df_zim['Frequency_Score'] * 0.35 +
    df_zim['Monetary_Score']  * 0.25
)

# 3.2 Zero balance (key Zimbabwe signal)
df_zim['Has_Zero_Balance']        = (df_zim['Balance'] == 0).astype(int)
df_zim['Balance_to_Salary_Ratio'] = (
    df_zim['Balance'] / (df_zim['EstimatedSalary'] + 1)
).round(4)

# 3.3 Account currency — ZWL vs USD (simulated)
np.random.seed(42)
df_zim['Account_Currency'] = np.where(
    (df_zim['Balance'] < 30000) & (np.random.rand(len(df_zim)) < 0.55),
    'ZWL', 'USD'
)

# Inflation-adjusted balance (ZWL ~500% annual inflation)
df_zim['Real_Balance_After_Inflation'] = np.where(
    df_zim['Account_Currency'] == 'ZWL',
    df_zim['Balance'] / 5.0,
    df_zim['Balance'] / 1.05
).round(2)

# 3.4 Mobile money competition risk
# Customers who are inactive + zero balance + single product
# are likely using EcoCash/OneMoney instead of the bank
df_zim['Mobile_Money_Risk'] = (
    (df_zim['IsActiveMember'] == 0).astype(int) +
    (df_zim['Has_Zero_Balance'] == 1).astype(int) +
    (df_zim['NumOfProducts'] == 1).astype(int)
)

# 3.5 Digital adoption score
df_zim['Digital_Adoption_Score'] = (
    df_zim['IsActiveMember'] * 2 +
    df_zim['HasCrCard']      * 1 +
    (df_zim['NumOfProducts'] >= 2).astype(int)
).clip(0, 4)

# 3.6 Age segment (for MicroFinance targeting)
df_zim['Age_Segment'] = pd.cut(
    df_zim['Age'],
    bins=[0, 25, 35, 50, 100],
    labels=['Youth (18-25)', 'Young Adult (26-35)',
            'Middle (36-50)', 'Senior (51+)']
)

# 3.7 Loyalty score
df_zim['Loyalty_Score'] = (
    df_zim['Tenure']         * 1.0 +
    df_zim['NumOfProducts']  * 0.5 +
    df_zim['IsActiveMember'] * 2.0
).round(2)

# ── Print key Zimbabwe findings ───────────────────────────────────────────────
print("\n[6] Mobile money risk vs churn rate:")
print(df_zim.groupby('Mobile_Money_Risk')['Exited']
      .agg(['mean', 'count'])
      .rename(columns={'mean': 'churn_rate', 'count': 'customers'})
      .assign(churn_rate=lambda x: x['churn_rate'].map('{:.1%}'.format))
      .to_string())

print("\n[7] Age segment churn rates:")
print(df_zim.groupby('Age_Segment', observed=True)['Exited']
      .agg(['mean', 'count'])
      .rename(columns={'mean': 'churn_rate', 'count': 'customers'})
      .assign(churn_rate=lambda x: x['churn_rate'].map('{:.1%}'.format))
      .to_string())

print("\n[8] Account currency split:")
print(df_zim['Account_Currency'].value_counts().to_string())

# ── Feature correlations ──────────────────────────────────────────────────────
eng_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'RFM_Score', 'Has_Zero_Balance', 'Balance_to_Salary_Ratio',
    'Mobile_Money_Risk', 'Digital_Adoption_Score', 'Loyalty_Score',
    'Exited'
]
print("\n[9] Feature correlations with churn (sorted):")
corr = df_zim[eng_cols].corr()['Exited'].drop('Exited').sort_values()
print(corr.round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 4 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[10] Generating plots ...")

plt.style.use('seaborn-v0_8-whitegrid')
C = {'churned': '#D85A30', 'retained': '#1D9E75'}

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    'Bank Customer Churn — EDA & Zimbabwe Feature Analysis',
    fontsize=16, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Donut: churn distribution
ax1 = fig.add_subplot(gs[0, 0])
sizes = df_zim['Exited'].value_counts()
ax1.pie(sizes, labels=['Retained', 'Churned'],
        autopct='%1.1f%%',
        colors=[C['retained'], C['churned']],
        startangle=90, pctdistance=0.75,
        wedgeprops={'width': 0.5})
ax1.set_title('Churn Distribution', fontweight='bold')

# Age histogram
ax2 = fig.add_subplot(gs[0, 1])
for label, color, name in [(0, C['retained'], 'Retained'),
                             (1, C['churned'],  'Churned')]:
    ax2.hist(df_zim[df_zim['Exited'] == label]['Age'],
             bins=20, alpha=0.6, color=color, label=name, density=True)
ax2.set_title('Age Distribution by Churn', fontweight='bold')
ax2.set_xlabel('Age'); ax2.set_ylabel('Density'); ax2.legend()

# Churn by num products
ax3 = fig.add_subplot(gs[0, 2])
prod_churn = df_zim.groupby('NumOfProducts')['Exited'].mean() * 100
ax3.bar(prod_churn.index, prod_churn.values,
        color=[C['churned'] if v > 20 else C['retained']
               for v in prod_churn.values])
ax3.set_title('Churn Rate by No. of Products', fontweight='bold')
ax3.set_xlabel('Number of Products'); ax3.set_ylabel('Churn Rate (%)')

# Mobile money risk (Zimbabwe)
ax4 = fig.add_subplot(gs[1, 0])
mm = df_zim.groupby('Mobile_Money_Risk')['Exited'].mean() * 100
ax4.bar(mm.index.astype(str), mm.values, color=C['churned'], alpha=0.8)
ax4.set_title('Mobile Money Risk vs Churn\n(Zimbabwe-specific)', fontweight='bold')
ax4.set_xlabel('Risk Score (0=low, 3=high)'); ax4.set_ylabel('Churn Rate (%)')
for i, v in enumerate(mm.values):
    ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

# RFM boxplot
ax5 = fig.add_subplot(gs[1, 1])
ax5.boxplot(
    [df_zim[df_zim['Exited'] == 0]['RFM_Score'],
     df_zim[df_zim['Exited'] == 1]['RFM_Score']],
    labels=['Retained', 'Churned'], patch_artist=True,
    boxprops={'facecolor': C['retained']}
)
ax5.set_title('RFM Score: Retained vs Churned', fontweight='bold')
ax5.set_ylabel('RFM Score')

# Digital adoption line
ax6 = fig.add_subplot(gs[1, 2])
da = df_zim.groupby('Digital_Adoption_Score')['Exited'].mean() * 100
ax6.plot(da.index, da.values, marker='o', color=C['churned'], linewidth=2)
ax6.fill_between(da.index, da.values, alpha=0.2, color=C['churned'])
ax6.set_title('Digital Adoption vs Churn', fontweight='bold')
ax6.set_xlabel('Digital Adoption Score'); ax6.set_ylabel('Churn Rate (%)')

# Correlation heatmap
ax7 = fig.add_subplot(gs[2, :2])
mask = np.triu(np.ones_like(df_zim[eng_cols].corr(), dtype=bool))
sns.heatmap(df_zim[eng_cols].corr(), mask=mask, ax=ax7,
            cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5)
ax7.set_title('Feature Correlation Matrix', fontweight='bold')
ax7.tick_params(axis='x', rotation=45)

# Age segment churn
ax8 = fig.add_subplot(gs[2, 2])
seg = df_zim.groupby('Age_Segment', observed=True)['Exited'].mean() * 100
bars = ax8.barh(seg.index.astype(str), seg.values,
                color=C['churned'], alpha=0.8)
ax8.set_title('Churn by Age Segment\n(MicroFinance targeting)', fontweight='bold')
ax8.set_xlabel('Churn Rate (%)')
for bar, v in zip(bars, seg.values):
    ax8.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{v:.1f}%', va='center', fontsize=9)

plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: outputs/eda_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 — SAVE ENGINEERED DATASET
# ═══════════════════════════════════════════════════════════════════════════════
final_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'RFM_Score', 'Has_Zero_Balance', 'Balance_to_Salary_Ratio',
    'Mobile_Money_Risk', 'Digital_Adoption_Score', 'Loyalty_Score',
    'Real_Balance_After_Inflation',
    'Geography', 'Gender', 'Account_Currency',
    'Exited'
]
df_zim[final_features].to_csv(OUT_CSV, index=False)
print(f"    Saved: data/churn_engineered.csv")

print(f"\n  Original features  : 11")
print(f"  Engineered features: 7  (Zimbabwe-adapted)")
print(f"  Total features     : {len(final_features) - 1} (excl. target)")

print("\n" + "=" * 60)
print(" DONE. Run step3_model.py next.")
print("=" * 60)