#!/usr/bin/env python3
"""
================================================================================
 元话语特征三实验分析 + Biber对比
 Metadiscourse Feature Analysis: Regression + Classification + Journal
================================================================================

一次性运行: 特征提取 → 回归(NCC) → 分类(高/低影响力) → 期刊分类 → Biber对比

输出: lab_Metadiscourse/output/ 中的所有图表和数据文件
================================================================================
"""
import warnings, sys, os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 10})

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

from metadiscourse_extractor import MetadiscourseExtractor

NCC_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"
FEATURE_NAMES = MetadiscourseExtractor.FEATURE_NAMES

print("=" * 70)
print(" 元话语特征分析: 回归 + 分类 + 期刊")
print(f" 特征数: {len(FEATURE_NAMES)}, 输出: {OUTPUT_DIR}")
print("=" * 70)

# ============================================================================
# 0. Feature Extraction
# ============================================================================
print("\n[0] 提取元话语特征...")
df_ncc = pd.read_csv(NCC_PATH)
extractor = MetadiscourseExtractor()
df_features = extractor.extract_all(df_ncc)
print(f"  Extracted: {df_features.shape}")

# Merge with NCC + journal data
df = df_features.merge(
    df_ncc[['doi', 'year', 'citations', 'NCC', 'source_journal']],
    left_on='doc_id', right_on='doi', how='inner'
).drop(columns=['doi'])
df = df.dropna(subset=['NCC'])

# Remove NCC outliers
ncc_upper, ncc_lower = df['NCC'].quantile(0.995), df['NCC'].quantile(0.005)
df = df[(df['NCC'] >= ncc_lower) & (df['NCC'] <= ncc_upper)].copy()
print(f"  Final dataset: {len(df)} abstracts")

# Save feature matrix
df_features.to_csv(OUTPUT_DIR / "feature_matrix.csv", index=False)
print(f"  Saved: feature_matrix.csv")

# Journal abbreviations
JOURNAL_ABBREV = {
    'Materials and Corrosion': 'Mat.Corros.',
    'CORROSION': 'CORROSION',
    'Anti-Corrosion Methods and Materials': 'Anti-Corr.MM.',
    'Corrosion Engineering Science and Technology The International Journal of Corrosion Processes and Corrosion Control': 'Corr.Eng.Sci.',
    'Corrosion Science': 'Corr.Sci.',
    'Corrosion and Materials Degradation': 'Corr.Mat.Deg.',
}
df['j_short'] = df['source_journal'].map(JOURNAL_ABBREV)

# Standardize features for modeling
scaler = StandardScaler()
X_all = scaler.fit_transform(df[FEATURE_NAMES].values)
y_ncc = df['NCC'].values
n_samples, n_features = X_all.shape

# ============================================================================
# 1. Descriptive Stats
# ============================================================================
print("\n[1] 描述性统计...")
desc = df[FEATURE_NAMES].describe()
print("  Top 5 features by mean:")
for f in desc.loc['mean'].sort_values(ascending=False).head().index:
    print(f"    {f:25s} mean={desc.loc['mean', f]:.4f}  std={desc.loc['std', f]:.4f}")

# Feature correlation matrix
feat_corr = df[FEATURE_NAMES].corr()

# ============================================================================
# 2. Experiment A: Regression (Features -> NCC)
# ============================================================================
print("\n[2] 实验A: 元话语特征 -> NCC 回归...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 回归用KFold
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 分类用Stratified

# Pearson correlations
pearson_res = {}
for feat in FEATURE_NAMES:
    r, p = pearsonr(df[feat], y_ncc)
    pearson_res[feat] = {'r': r, 'p': p}
top_feat = max(pearson_res, key=lambda k: abs(pearson_res[k]['r']))

# OLS
ols = LinearRegression().fit(X_all, y_ncc)
y_pred = ols.predict(X_all)
ss_res, ss_tot = np.sum((y_ncc - y_pred)**2), np.sum((y_ncc - np.mean(y_ncc))**2)
r2_ols = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2_ols) * (n_samples - 1) / (n_samples - n_features - 1)

# Ridge CV
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=3).fit(X_all, y_ncc)
ridge_cv = cross_val_score(Ridge(alpha=ridge.alpha_), X_all, y_ncc, cv=cv, scoring='r2')

# Lasso CV
lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=3, max_iter=10000, random_state=42).fit(X_all, y_ncc)
lasso_cv = cross_val_score(Lasso(alpha=lasso.alpha_, max_iter=10000, random_state=42), X_all, y_ncc, cv=cv, scoring='r2')

# RF
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=50, random_state=42, n_jobs=-1)
rf_reg.fit(X_all, y_ncc)
rf_cv = cross_val_score(rf_reg, X_all, y_ncc, cv=cv, scoring='r2')

# Feature importance
rf_importance = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': rf_reg.feature_importances_}).sort_values('importance', ascending=False)

# Cohen's d per feature (Q4 vs Q1)
cohens_res = {}
for feat in FEATURE_NAMES:
    q1_mask = df[feat] <= df[feat].quantile(0.25)
    q4_mask = df[feat] >= df[feat].quantile(0.75)
    g1, g2 = df[q1_mask]['NCC'].values, df[q4_mask]['NCC'].values
    n1, n2 = len(g1), len(g2)
    sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    d = (np.mean(g2)-np.mean(g1))/max(sp, 1e-10) if sp > 1e-10 else 0
    _, p = stats.ttest_ind(g2, g1)
    cohens_res[feat] = {'d': d, 'p': p}

print(f"  Top Pearson |r|: {top_feat} r={pearson_res[top_feat]['r']:+.4f}")
print(f"  OLS R^2={r2_ols:.4f} AdjR^2={adj_r2:.4f}")
print(f"  Ridge CV R^2={ridge_cv.mean():.4f}+/-{ridge_cv.std():.4f}")
print(f"  Lasso CV R^2={lasso_cv.mean():.4f}+/-{lasso_cv.std():.4f}")
print(f"  RF CV R^2={rf_cv.mean():.4f}+/-{rf_cv.std():.4f}")

# ============================================================================
# 3. Experiment B: High/Low Impact Classification
# ============================================================================
print("\n[3] 实验B: 元话语特征 -> 高/低影响力分类...")

class_results = {}
strat_configs = [
    ('S1_Top25vsBot25', 0.75, 0.25),
    ('S2_Top20vsBot20', 0.80, 0.20),
    ('S4_Top10vsBot10', 0.90, 0.10),
]
for strat_name, hi_q, lo_q in strat_configs:
    th_hi = df['NCC'].quantile(hi_q)
    th_lo = df['NCC'].quantile(lo_q)
    subset = df[(df['NCC'] >= th_hi) | (df['NCC'] <= th_lo)].copy()
    y_s = (subset['NCC'] >= th_hi).astype(int)

    if y_s.sum() < 10 or (len(y_s)-y_s.sum()) < 10 or len(subset) < 100:
        continue

    X_s = StandardScaler().fit_transform(subset[FEATURE_NAMES].values)
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
    n_cv = min(5, y_s.sum(), len(y_s)-y_s.sum())
    use_cv = StratifiedKFold(n_splits=max(3, n_cv), shuffle=True, random_state=42) if n_cv >= 3 else KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(rf_clf, X_s, y_s, cv=use_cv, scoring={'auc': 'roc_auc', 'f1': 'f1'}, n_jobs=-1)
    class_results[strat_name] = {
        'auc': scores['test_auc'].mean(), 'auc_std': scores['test_auc'].std(),
        'f1': scores['test_f1'].mean(),
        'n': len(subset), 'baseline': max(y_s.mean(), 1-y_s.mean()),
    }
    print(f"  {strat_name}: AUC={class_results[strat_name]['auc']:.3f}+/-{class_results[strat_name]['auc_std']:.3f}")

# ============================================================================
# 4. Experiment C: Journal Classification
# ============================================================================
print("\n[4] 实验C: 元话语特征 -> 期刊分类...")

journals = sorted(df['source_journal'].unique())
j_short = [JOURNAL_ABBREV[j] for j in journals]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_j = le.fit_transform(df['source_journal'])
n_j = len(journals)

# Multi-class
rf_mc = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=30, random_state=42, n_jobs=-1)
mc_scores = cross_validate(rf_mc, X_all, y_j, cv=cv_strat, scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}, n_jobs=-1)
baseline_mc = max(np.bincount(y_j))/len(y_j)

# One-vs-Rest
rf_mc.fit(X_all, y_j)
ovr_results = {}
for i, jn in enumerate(journals):
    y_bin = (y_j == i).astype(int)
    rf_ovr = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
    ovr_scores = cross_validate(rf_ovr, X_all, y_bin, cv=cv_strat, scoring={'auc': 'roc_auc', 'f1': 'f1'}, n_jobs=-1)
    ovr_results[JOURNAL_ABBREV[jn]] = {
        'auc': ovr_scores['test_auc'].mean(), 'auc_std': ovr_scores['test_auc'].std(),
        'f1': ovr_scores['test_f1'].mean(),
        'n': (y_j==i).sum(),
    }

# Confusion matrix
from sklearn.model_selection import cross_val_predict
y_pred_cv = cross_val_predict(rf_mc, X_all, y_j, cv=cv_strat, n_jobs=-1)
cm = confusion_matrix(y_j, y_pred_cv)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Pairwise Cohen's d
pairwise_d = []
for j1, j2 in combinations(range(n_j), 2):
    d1, d2 = df[df['source_journal']==journals[j1]], df[df['source_journal']==journals[j2]]
    for feat in ['hedge_ratio', 'booster_ratio', 'transition_ratio', 'self_mention_ratio']:
        n1, n2 = len(d1), len(d2)
        sp = np.sqrt(((n1-1)*np.var(d1[feat],ddof=1)+(n2-1)*np.var(d2[feat],ddof=1))/(n1+n2-2))
        d_val = (np.mean(d1[feat])-np.mean(d2[feat]))/max(sp, 1e-10)
        pairwise_d.append({'j1': JOURNAL_ABBREV[journals[j1]], 'j2': JOURNAL_ABBREV[journals[j2]], 'feat': feat, 'd': abs(d_val)})
pairwise_d.sort(key=lambda x: x['d'], reverse=True)

# Per-journal feature profiles
j_profiles = {}
for jn, js in zip(journals, j_short):
    jdf = df[df['source_journal']==jn]
    j_profiles[js] = {feat: (jdf[feat].mean()-df[feat].mean())/df[feat].std() for feat in FEATURE_NAMES}

print(f"  Multi-class: Acc={mc_scores['test_accuracy'].mean():.3f} (baseline={baseline_mc:.3f})")
print(f"  F1_macro={mc_scores['test_f1_macro'].mean():.3f}")
print(f"  Top OvR AUC: {max(ovr_results.items(), key=lambda x: x[1]['auc'])[0]} AUC={max(v['auc'] for v in ovr_results.values()):.3f}")
print(f"  Top pairwise d: {pairwise_d[0]['j1']} vs {pairwise_d[0]['j2']} ({pairwise_d[0]['feat']}) d={pairwise_d[0]['d']:.3f}")

# ============================================================================
# 5. Comparison with Biber Results
# ============================================================================
print("\n[5] Biber对比...")

biber_results = {
    'regression_r2': 0.0063,
    'regression_top_r': 0.059,
    'class_best_auc': 0.604,
    'journal_acc': 0.332,
    'journal_baseline': 0.313,
    'journal_best_ovr_auc': 0.751,
    'journal_f1_macro': 0.173,
}

meta_best_auc = max(v['auc'] for v in class_results.values()) if class_results else 0
meta_best_ovr = max(v['auc'] for v in ovr_results.values()) if ovr_results else 0

comparison = pd.DataFrame({
    'Metric': [
        '回归 R^2 (OLS)',
        '回归 最强Pearson |r|',
        '分类 最佳AUC',
        '期刊 多分类Acc',
        '期刊 Baseline',
        '期刊 最佳OvR AUC',
        '期刊 F1_macro',
    ],
    'Biber (15特征+4PCA)': [
        f"{biber_results['regression_r2']:.4f}",
        f"{biber_results['regression_top_r']:.3f}",
        f"{biber_results['class_best_auc']:.3f}",
        f"{biber_results['journal_acc']:.3f}",
        f"{biber_results['journal_baseline']:.3f}",
        f"{biber_results['journal_best_ovr_auc']:.3f}",
        f"{biber_results['journal_f1_macro']:.3f}",
    ],
    'Metadiscourse (10特征)': [
        f"{r2_ols:.4f}",
        f"{max(abs(v['r']) for v in pearson_res.values()):.3f}",
        f"{meta_best_auc:.3f}",
        f"{mc_scores['test_accuracy'].mean():.3f}",
        f"{baseline_mc:.3f}",
        f"{meta_best_ovr:.3f}",
        f"{mc_scores['test_f1_macro'].mean():.3f}",
    ],
})

print(comparison.to_string(index=False))

# Determine winner per category
reg_winner = 'Metadiscourse' if r2_ols > biber_results['regression_r2'] else 'Biber'
cls_winner = 'Metadiscourse' if meta_best_auc > biber_results['class_best_auc'] else 'Biber'
jrn_winner = 'Metadiscourse' if mc_scores['test_accuracy'].mean() > biber_results['journal_acc'] else 'Biber'
print(f"\n  Regression winner: {reg_winner}")
print(f"  Classification winner: {cls_winner}")
print(f"  Journal winner: {jrn_winner}")

# ============================================================================
# 6. Visualizations
# ============================================================================
print("\n[6] 生成图表...")

# Fig 1: Feature importance (RF regression)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
rf_imp_sorted = rf_importance.sort_values('importance')
axes[0].barh(rf_imp_sorted['feature'], rf_imp_sorted['importance'], color='forestgreen', edgecolor='white')
axes[0].set_title('RF Feature Importance (NCC Regression)')
axes[0].set_xlabel('Importance')

# Top features by |correlation| with NCC
corr_vals = [(f, pearson_res[f]['r']) for f in FEATURE_NAMES]
corr_vals.sort(key=lambda x: abs(x[1]))
axes[1].barh([c[0] for c in corr_vals], [c[1] for c in corr_vals],
             color=['#d73027' if v<0 else '#4575b4' for _, v in corr_vals], edgecolor='white')
axes[1].axvline(0, color='black', linewidth=0.5)
axes[1].set_title('Pearson r with NCC')
axes[1].set_xlabel('Correlation coefficient')
fig.suptitle('Metadiscourse Feature Analysis: NCC Prediction', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_m1_feature_importance.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_m1_feature_importance.png")

# Fig 2: Journal OVR AUC comparison
fig, ax = plt.subplots(figsize=(10, 5))
ovr_items = sorted(ovr_results.items(), key=lambda x: x[1]['auc'], reverse=True)
names, aucs, stds = zip(*[(k, v['auc'], v['auc_std']) for k, v in ovr_items])
colors = ['#d73027' if a > 0.7 else '#fc8d59' if a > 0.6 else '#4575b4' for a in aucs]
ax.barh(range(len(names)), aucs, xerr=stds, color=colors, edgecolor='white', linewidth=1.5, capsize=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.axvline(0.5, color='gray', linewidth=0.8, linestyle='--', label='Random (0.5)')
ax.set_xlabel('AUC-ROC (5-fold CV)')
ax.set_title('One-vs-Rest Journal Classification: Metadiscourse Features', fontsize=14)
for i, (name, auc) in enumerate(zip(names, aucs)):
    ax.text(auc + 0.005, i, f'{auc:.3f}', va='center', fontsize=9)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_m2_journal_ovr_auc.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_m2_journal_ovr_auc.png")

# Fig 3: Confusion matrix
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=j_short, yticklabels=j_short,
            cmap='YlOrRd', vmin=0, vmax=0.5, linewidths=1,
            cbar_kws={'label': 'Recall Rate', 'shrink': 0.8}, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Journal Confusion Matrix (Metadiscourse)\nAcc={mc_scores["test_accuracy"].mean():.3f} F1={mc_scores["test_f1_macro"].mean():.3f}', fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_m3_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_m3_confusion_matrix.png")

# Fig 4: Journal profiles (top 4 features)
fig, ax = plt.subplots(figsize=(10, 6))
top4_feats = [c[0] for c in corr_vals[-4:]]  # 4 features with highest |r|
x = np.arange(len(top4_feats))
colors_j = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
for i, (js, color) in enumerate(zip(j_short, colors_j)):
    profile = [j_profiles[js][f] for f in top4_feats]
    ax.plot(x, profile, 'o-', color=color, linewidth=2, markersize=8, label=js)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(top4_feats, fontsize=11)
ax.set_ylabel('Standardized Deviation from Grand Mean')
ax.set_title('Journal Metadiscourse Profiles (Top 4 NCC-correlated Features)', fontsize=14)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_m4_journal_profiles.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_m4_journal_profiles.png")

# Fig 5: Biber vs Metadiscourse comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Regression R^2
axes[0].bar(['Biber\n(15+4PCA)', 'Metadiscourse\n(10 raw)'],
            [biber_results['regression_r2'], r2_ols],
            color=['#4575b4', '#d73027'], edgecolor='white', linewidth=2)
axes[0].set_title('Regression R^2 (NCC Prediction)')
for i, v in enumerate([biber_results['regression_r2'], r2_ols]):
    axes[0].text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')

# Classification AUC
axes[1].bar(['Biber', 'Metadiscourse'],
            [biber_results['class_best_auc'], meta_best_auc],
            color=['#4575b4', '#d73027'], edgecolor='white', linewidth=2)
axes[1].axhline(0.5, color='gray', linewidth=0.8, linestyle='--', label='Random')
axes[1].set_title('Best Classification AUC')
for i, v in enumerate([biber_results['class_best_auc'], meta_best_auc]):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

# Journal accuracy
axes[2].bar(['Biber', 'Metadiscourse'],
            [biber_results['journal_acc'] - biber_results['journal_baseline'],
             mc_scores['test_accuracy'].mean() - baseline_mc],
            color=['#4575b4', '#d73027'], edgecolor='white', linewidth=2)
axes[2].axhline(0, color='black', linewidth=0.5)
axes[2].set_title('Journal Acc Above Baseline')
for i, v in enumerate([biber_results['journal_acc'] - biber_results['journal_baseline'],
                        mc_scores['test_accuracy'].mean() - baseline_mc]):
    axes[2].text(i, v + 0.002 if v >= 0 else v - 0.008, f'{v:+.3f}', ha='center', fontsize=12, fontweight='bold')

fig.suptitle('Biber MD vs. Metadiscourse: Head-to-Head Comparison', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_m5_biber_vs_meta.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_m5_biber_vs_meta.png")

# ============================================================================
# 7. Summary
# ============================================================================
print("\n[7] 分析总结...")

# Find best single feature
best_feat = max(pearson_res, key=lambda k: abs(pearson_res[k]['r']))
print(f"\n  最佳单特征: {best_feat} (r={pearson_res[best_feat]['r']:+.4f}, p={pearson_res[best_feat]['p']:.6f})")

# Feature with strongest journal discrimination
best_j_feat = pairwise_d[0]['feat'] if pairwise_d else 'N/A'
print(f"  最佳期刊区分特征: {best_j_feat} (最强对d={pairwise_d[0]['d']:.3f})" if pairwise_d else "")

# Practical recommendation
print(f"\n  对Copilot的启示:")
top3 = sorted(pearson_res.items(), key=lambda x: abs(x[1]['r']), reverse=True)[:3]
for feat, res in top3:
    direction = 'higher' if res['r'] > 0 else 'lower'
    print(f"    {feat}: {direction} values weakly associated with higher NCC (r={res['r']:+.3f})")

print(f"\n  Biber vs Metadiscourse:")
improvements = []
for metric, biber_val, meta_val in [
    ('Regression R^2', biber_results['regression_r2'], r2_ols),
    ('Classification AUC', biber_results['class_best_auc'], meta_best_auc),
    ('Journal Acc-Baseline', biber_results['journal_acc']-biber_results['journal_baseline'], mc_scores['test_accuracy'].mean()-baseline_mc),
]:
    delta = meta_val - biber_val
    improvements.append((metric, delta > 0, delta))
    direction = '>' if delta > 0 else '<'
    print(f"    {metric}: Metadiscourse {direction} Biber (delta={delta:+.4f})")

if sum(1 for _, better, _ in improvements if better) >= 2:
    print(f"\n  Metadiscourse outperforms Biber on most metrics.")
else:
    print(f"\n  Biber and Metadiscourse are comparable — both show limited signal.")

# Save results summary
summary = {
    'n_samples': n_samples, 'n_features': n_features,
    'regression_r2_ols': r2_ols, 'regression_adj_r2': adj_r2,
    'regression_ridge_cv_r2': ridge_cv.mean(), 'regression_lasso_cv_r2': lasso_cv.mean(),
    'regression_rf_cv_r2': rf_cv.mean(),
    'best_pearson_r': pearson_res[best_feat]['r'], 'best_pearson_feature': best_feat,
    'classification_best_auc': meta_best_auc,
    'journal_accuracy': mc_scores['test_accuracy'].mean(),
    'journal_baseline': baseline_mc,
    'journal_f1_macro': mc_scores['test_f1_macro'].mean(),
    'journal_best_ovr_auc': meta_best_ovr,
}
pd.DataFrame([summary]).to_csv(OUTPUT_DIR / 'results_summary.csv', index=False)

print(f"\n  [OK] 分析完成! 所有结果保存至: {OUTPUT_DIR}")
print("=" * 70)
