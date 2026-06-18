"""
BANK CHURN PREDICTION PROJECT
Zimbabwe Banking/MicroFinance — Industrial Attachment

Step 3: Model Training, Evaluation & SHAP Explainability
=========================================================
Reads:  data/churn_engineered.csv
Writes: models/churn_model_bundle.pkl
        outputs/model_evaluation.png
        outputs/shap_analysis.png
        outputs/churn_risk_scores.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data',    'churn_engineered.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models',  'churn_model_bundle.pkl')
EVAL_PLOT  = os.path.join(BASE_DIR, 'outputs', 'model_evaluation.png')
SHAP_PLOT  = os.path.join(BASE_DIR, 'outputs', 'shap_analysis.png')
RISK_CSV   = os.path.join(BASE_DIR, 'outputs', 'churn_risk_scores.csv')
os.makedirs(os.path.join(BASE_DIR, 'models'),  exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — LOAD & PREPROCESS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(" STEP 3 — MODEL TRAINING & EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded: {len(df):,} rows | Churn rate: {df['Exited'].mean()*100:.1f}%")

# Encode categorical columns
encoders = {}
for col in ['Geography', 'Gender', 'Account_Currency']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

TARGET   = 'Exited'
FEATURES = [c for c in df.columns if c != TARGET]
X, y     = df[FEATURES], df[TARGET]

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[2] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"    Train churn rate: {y_train.mean()*100:.1f}%")
print(f"    Test  churn rate: {y_test.mean()*100:.1f}%")

# Scale features
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# SMOTE — only on training data, never on test
smote        = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"\n[3] After SMOTE: {len(X_train_sm):,} samples "
      f"(balanced {y_train_sm.mean()*100:.0f}% / {(1-y_train_sm.mean())*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2 — TRAIN FOUR MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" TRAINING MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=RANDOM_STATE
    ),
}

results = {}
for name, model in models.items():
    print(f"\n  Training: {name} ...")

    # XGBoost handles imbalance via scale_pos_weight — no SMOTE needed
    X_tr = X_train_sc if name == 'XGBoost' else X_train_sm
    y_tr = y_train    if name == 'XGBoost' else y_train_sm

    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    auc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred)
    ap     = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    cv_auc = cross_val_score(
        model, X_train_sc, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc', n_jobs=-1
    ).mean()

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'auc': auc, 'cv_auc': cv_auc, 'f1': f1,
        'ap': ap, 'report': report, 'cm': cm,
    }

    prec = report['1']['precision']
    rec  = report['1']['recall']
    print(f"    AUC-ROC   : {auc:.4f}  (5-fold CV: {cv_auc:.4f})")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    Precision : {prec:.4f}  |  Recall: {rec:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" MODEL COMPARISON")
print("=" * 60)

rows = []
for name, r in results.items():
    rep = r['report']
    rows.append({
        'Model':         name,
        'AUC-ROC':       round(r['auc'],    4),
        'CV-AUC':        round(r['cv_auc'], 4),
        'F1 (churn)':    round(r['f1'],     4),
        'Precision':     round(rep['1']['precision'], 4),
        'Recall':        round(rep['1']['recall'],    4),
        'Avg Precision': round(r['ap'],     4),
    })

compare_df = pd.DataFrame(rows).set_index('Model')
print(compare_df.to_string())

best_name  = compare_df['AUC-ROC'].idxmax()
best_model = results[best_name]['model']
print(f"\n  Best model: {best_name}  (AUC={results[best_name]['auc']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 4 — SAVE MODEL BUNDLE
# ═══════════════════════════════════════════════════════════════════════════════
joblib.dump({
    'model':    best_model,
    'scaler':   scaler,
    'features': FEATURES,
    'encoders': encoders,
    'best_model_name': best_name,
}, MODEL_PATH)
print(f"\n[4] Model bundle saved → models/churn_model_bundle.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 — EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Generating evaluation plots ...")

PALETTE = {
    'Logistic Regression': '#534AB7',
    'Random Forest':       '#1D9E75',
    'XGBoost':             '#D85A30',
    'Gradient Boosting':   '#BA7517',
}

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    'Churn Model Evaluation — Zimbabwe Banking/MicroFinance',
    fontsize=16, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ROC curves
ax1 = fig.add_subplot(gs[0, 0])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax1.plot(fpr, tpr, label=f"{name} ({r['auc']:.3f})",
             color=PALETTE[name], linewidth=1.8)
ax1.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves', fontweight='bold')
ax1.legend(fontsize=7.5, loc='lower right')

# Precision-Recall curves
ax2 = fig.add_subplot(gs[0, 1])
for name, r in results.items():
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, r['y_prob'])
    ax2.plot(rec_arr, prec_arr,
             label=f"{name} (AP={r['ap']:.3f})",
             color=PALETTE[name], linewidth=1.8)
ax2.axhline(y_test.mean(), linestyle='--', color='grey',
            linewidth=1, label=f"Baseline ({y_test.mean():.3f})")
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves', fontweight='bold')
ax2.legend(fontsize=7.5)

# Metric bar chart
ax3 = fig.add_subplot(gs[0, 2])
metrics = ['AUC-ROC', 'F1 (churn)', 'Precision', 'Recall']
x = np.arange(len(metrics))
width = 0.18
for i, (name, r) in enumerate(results.items()):
    rep  = r['report']
    vals = [r['auc'], r['f1'],
            rep['1']['precision'], rep['1']['recall']]
    ax3.bar(x + i * width, vals, width,
            label=name, color=PALETTE[name], alpha=0.85)
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels(metrics, fontsize=8)
ax3.set_ylim(0, 1.05)
ax3.set_title('Metric Comparison', fontweight='bold')
ax3.legend(fontsize=7, loc='lower right')
ax3.set_ylabel('Score')

# Confusion matrix (best model)
ax4 = fig.add_subplot(gs[1, 0])
cm = results[best_name]['cm']
ax4.imshow(cm, interpolation='nearest', cmap='Blues')
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Retained', 'Churned'])
ax4.set_yticklabels(['Retained', 'Churned'])
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax4.set_xlabel('Predicted'); ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')

# Feature importance (best model)
ax5 = fig.add_subplot(gs[1, 1])
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
top15    = feat_imp.tail(15)
zim_keys = ['Mobile_Money', 'Digital', 'RFM', 'Account', 'Real_Balance', 'Loyalty']
colors   = ['#D85A30' if any(k in f for k in zim_keys) else '#534AB7'
            for f in top15.index]
top15.plot(kind='barh', ax=ax5, color=colors)
ax5.set_title(f'Top 15 Feature Importances\n({best_name})', fontweight='bold')
ax5.set_xlabel('Importance')
legend_el = [
    mpatches.Patch(facecolor='#D85A30', label='Zimbabwe-engineered'),
    mpatches.Patch(facecolor='#534AB7', label='Original feature'),
]
ax5.legend(handles=legend_el, fontsize=8, loc='lower right')

# Churn probability distribution
ax6 = fig.add_subplot(gs[1, 2])
best_probs = results[best_name]['y_prob']
ax6.hist(best_probs[y_test == 0], bins=40, alpha=0.6,
         color='#1D9E75', label='Retained', density=True)
ax6.hist(best_probs[y_test == 1], bins=40, alpha=0.6,
         color='#D85A30', label='Churned',  density=True)
ax6.axvline(0.5, color='black', linestyle='--', linewidth=1,
            label='Threshold (0.5)')
ax6.set_xlabel('Predicted Churn Probability')
ax6.set_ylabel('Density')
ax6.set_title(f'Probability Distribution\n({best_name})', fontweight='bold')
ax6.legend(fontsize=8)

plt.savefig(EVAL_PLOT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: outputs/model_evaluation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[6] SHAP explainability ({best_name}) ...")

if best_name in ['XGBoost', 'Random Forest', 'Gradient Boosting']:
    explainer  = shap.TreeExplainer(best_model)
    shap_vals  = explainer.shap_values(X_test_sc)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
else:
    explainer = shap.LinearExplainer(best_model, X_train_sm)
    sv        = explainer.shap_values(X_test_sc)

shap_df       = pd.DataFrame(sv, columns=FEATURES)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

print("\n  Mean |SHAP| — top 10 features:")
print(mean_abs_shap.head(10).round(4).to_string())

fig_shap, axes = plt.subplots(1, 2, figsize=(16, 7))
fig_shap.suptitle(
    f'SHAP Explainability — {best_name}\nZimbabwe Banking Churn Model',
    fontsize=14, fontweight='bold'
)

# Left: mean |SHAP| bar
top_n      = 15
top_feats  = mean_abs_shap.head(top_n)
bar_colors = ['#D85A30' if any(k in f for k in zim_keys)
              else '#534AB7' for f in top_feats.index]
axes[0].barh(range(top_n), top_feats.values[::-1], color=bar_colors[::-1])
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_feats.index[::-1], fontsize=9)
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].set_title('Global Feature Importance (SHAP)', fontweight='bold')
legend_el = [
    mpatches.Patch(facecolor='#D85A30', label='Zimbabwe-engineered'),
    mpatches.Patch(facecolor='#534AB7', label='Original feature'),
]
axes[0].legend(handles=legend_el, fontsize=8)

# Right: beeswarm (manual)
top10     = mean_abs_shap.head(10).index.tolist()
sv_top10  = pd.DataFrame(sv, columns=FEATURES)[top10]
Xt_top10  = pd.DataFrame(X_test_sc, columns=FEATURES)[top10]

ax_r = axes[1]
for yi, feat in enumerate(reversed(top10)):
    sv_col  = sv_top10[feat].values
    fv_col  = Xt_top10[feat].values
    fv_norm = (fv_col - fv_col.min()) / (fv_col.max() - fv_col.min() + 1e-9)
    jitter  = np.random.RandomState(42).uniform(-0.25, 0.25, len(sv_col))
    sc      = ax_r.scatter(sv_col, yi + jitter,
                           c=fv_norm, cmap='RdYlGn_r',
                           s=4, alpha=0.5, linewidths=0)

ax_r.set_yticks(range(10))
ax_r.set_yticklabels(list(reversed(top10)), fontsize=9)
ax_r.axvline(0, color='grey', linewidth=0.8, linestyle='--')
ax_r.set_xlabel('SHAP value  (positive = higher churn risk)')
ax_r.set_title('SHAP Beeswarm — top 10 features\n'
               '(red=high feature value, green=low)',
               fontweight='bold')
plt.colorbar(sc, ax=ax_r, label='Feature value (normalised)', shrink=0.6)

plt.tight_layout()
plt.savefig(SHAP_PLOT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: outputs/shap_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7 — RISK SEGMENTATION CSV
# ═══════════════════════════════════════════════════════════════════════════════
risk_df = X_test.copy().reset_index(drop=True)
risk_df['Churn_Probability'] = best_probs
risk_df['Actual_Churn']      = y_test.reset_index(drop=True)
risk_df['Risk_Band']         = pd.cut(
    risk_df['Churn_Probability'],
    bins=[0, 0.30, 0.50, 0.70, 1.0],
    labels=['Low (<30%)', 'Medium (30-50%)', 'High (50-70%)', 'Critical (>70%)']
)

print("\n[7] Risk band distribution:")
print(risk_df.groupby('Risk_Band', observed=True).agg(
    customers=('Churn_Probability', 'count'),
    avg_prob=('Churn_Probability', 'mean'),
    actual_churn_rate=('Actual_Churn', 'mean')
).round(3).to_string())

risk_df.to_csv(RISK_CSV, index=False)
print(f"\n    Saved: outputs/churn_risk_scores.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 8 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" RESULTS SUMMARY")
print("=" * 60)
print(compare_df.to_string())
print(f"\n  Best model : {best_name}")
print(f"  AUC-ROC    : {results[best_name]['auc']:.4f}")
print(f"  F1 (churn) : {results[best_name]['f1']:.4f}")
print(f"\n  Files saved:")
print(f"    models/churn_model_bundle.pkl")
print(f"    outputs/model_evaluation.png")
print(f"    outputs/shap_analysis.png")
print(f"    outputs/churn_risk_scores.csv")
print("\n" + "=" * 60)
print(" DONE. Run step4_api.py next.")
print("=" * 60)