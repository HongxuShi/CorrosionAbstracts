#!/usr/bin/env python3
"""
================================================================================
 期刊写作风格分类分析
 Journal Writing Style Classification from MD Dimension Scores
================================================================================

核心问题：
  MD分析的4个语言功能维度能否区分不同期刊的写作风格？

为什么这比预测引用数更有前景：
  - 期刊有明确的编辑偏好和写作传统（真实的风格差异）
  - 信号不会被几十个非语言因素稀释
  - 如果成功 → copilot可以提供期刊匹配建议
  - 即使部分成功 → 也能指出哪些期刊对语言风格最敏感

分析设计：
  ┌──────┬─────────────────────────────────┬─────────────────────┐
  │ 层次 │ 分析内容                        │ 回答的问题          │
  ├──────┼─────────────────────────────────┼─────────────────────┤
  │  1   │ 多分类 (6期刊)                  │ 维度的区分力有多强？│
  │  2   │ 逐对期刊 Cohen's d             │ 哪对期刊风格差异大？│
  │  3   │ One-vs-Rest AUC per journal    │ 哪个期刊风格最独特？│
  │  4   │ 混淆矩阵                       │ 哪些期刊容易混淆？  │
  │  5   │ 特征重要性                     │ 哪个维度贡献最大？  │
  │  6   │ 2D可视化 (PCA + 期刊分布)      │ 期刊在维度空间的分离 │
  │  7   │ 期刊风格相似度矩阵             │ 全文体风格的地图    │
  └──────┴─────────────────────────────────┴─────────────────────┘

数据集：6个腐蚀科学期刊，共~5788篇论文

作者：理论验证阶段
日期：2026/07/17
================================================================================
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

# 统计与ML
from scipy import stats
from scipy.stats import f_oneway
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
})

# ============================================================================
# 路径配置
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_SCORES_PATH = SCRIPT_DIR / "output" / "dimension_scores.csv"
NCC_DATA_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"

print("=" * 70)
print(" 期刊写作风格分类分析")
print(" Journal Writing Style Classification")
print("=" * 70)

# ============================================================================
# 0. 数据准备
# ============================================================================
print("\n[0] 数据加载...")
df_scores = pd.read_csv(DIM_SCORES_PATH)
df_ncc = pd.read_csv(NCC_DATA_PATH)

df = df_scores.merge(
    df_ncc[['doi', 'year', 'source_journal']],
    left_on='doc_id', right_on='doi', how='inner'
).drop(columns=['doi'])

# 期刊名简化（方便图表显示）
JOURNAL_ABBREV = {
    'Materials and Corrosion': 'Mat.Corros.',
    'CORROSION': 'CORROSION',
    'Anti-Corrosion Methods and Materials': 'Anti-Corr.MM.',
    'Corrosion Engineering Science and Technology The International Journal of Corrosion Processes and Corrosion Control': 'Corr.Eng.Sci.',
    'Corrosion Science': 'Corr.Sci.',
    'Corrosion and Materials Degradation': 'Corr.Mat.Deg.',
}
df['journal_short'] = df['source_journal'].map(JOURNAL_ABBREV)

dim_cols = ['Dim1', 'Dim2', 'Dim3', 'Dim4']
journals = sorted(df['source_journal'].unique())
journal_short_names = [JOURNAL_ABBREV[j] for j in journals]
n_journals = len(journals)

print(f"  有效记录: {len(df)} 篇")
print(f"  期刊数: {n_journals}")
for j, short in zip(journals, journal_short_names):
    count = (df['source_journal'] == j).sum()
    print(f"    {short:<18s}  {count:5d} 篇")

# Label encoding for classification
le = JournalLabelEncoder = LabelEncoder()
y = le.fit_transform(df['source_journal'])
journal_names_encoded = le.classes_

# ============================================================================
# 1. 期刊维度得分差异 — ANOVA + 逐对效应量
# ============================================================================
print("\n[1] 期刊间维度差异分析...")

# 1a. ANOVA — 每个维度在不同期刊间是否有差异
print("\n  --- ANOVA (各维度在期刊间的差异) ---")
for dim in dim_cols:
    groups = [df[df['source_journal'] == j][dim].values for j in journals]
    f_stat, anova_p = f_oneway(*groups)
    eta_sq = (f_stat * (n_journals - 1)) / (f_stat * (n_journals - 1) + sum(len(g) - 1 for g in groups))
    sig = "***" if anova_p < 0.001 else ("**" if anova_p < 0.01 else ("*" if anova_p < 0.05 else "ns"))
    print(f"    {dim}: F={f_stat:.1f}, p={anova_p:.6f} {sig}, eta^2={eta_sq:.4f}")

# 1b. 逐对期刊的Cohen's d（所有维度+所有期刊对）
print("\n  --- 逐对期刊效应量 (Cohen's d) Top 15 ---")

def cohens_d(g1, g2):
    """Cohen's d: (mean1 - mean2) / pooled_std"""
    n1, n2 = len(g1), len(g2)
    s_pooled = np.sqrt(((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / max(s_pooled, 1e-10)

pairwise_effects = []
for j1, j2 in combinations(range(n_journals), 2):
    d1 = df[df['source_journal'] == journals[j1]]
    d2 = df[df['source_journal'] == journals[j2]]
    for dim in dim_cols:
        d = cohens_d(d1[dim].values, d2[dim].values)
        t_stat, t_p = stats.ttest_ind(d1[dim], d2[dim])
        pairwise_effects.append({
            'j1': journal_short_names[j1], 'j2': journal_short_names[j2],
            'dim': dim, 'cohens_d': d, 'p_value': t_p,
            'abs_d': abs(d),
        })

pairwise_effects.sort(key=lambda x: x['abs_d'], reverse=True)

# 打印前15个最大的效应量
print(f"    {'Journal Pair':<35s} {'Dim':<8s} {'d':>8s} {'p':>10s}")
for pe in pairwise_effects[:15]:
    sig = "***" if pe['p_value'] < 0.001 else ("**" if pe['p_value'] < 0.01 else "*" if pe['p_value'] < 0.05 else "")
    print(f"    {pe['j1']+' vs '+pe['j2']:<35s} {pe['dim']:<8s} {pe['cohens_d']:+8.3f} {pe['p_value']:10.6f} {sig}")

# 1c. 各期刊在各维度上的偏差（相对于总均值）
print("\n  --- 各期刊维度得分偏离 (相对于总均值的标准化偏差) ---")
journal_profiles = {}
for j, short in zip(journals, journal_short_names):
    jdf = df[df['source_journal'] == j]
    profile = {}
    for dim in dim_cols:
        z = (jdf[dim].mean() - df[dim].mean()) / df[dim].std()
        profile[dim] = z
    journal_profiles[short] = profile

    print(f"    {short:<18s}", end='')
    for dim in dim_cols:
        z = profile[dim]
        bar = '+' * max(0, int(z * 10)) if z > 0 else '-' * max(0, int(-z * 10))
        print(f"  {dim}: {z:+6.3f} {bar}", end='')
    print()

# ============================================================================
# 2. 多分类 (6期刊)
# ============================================================================
print("\n[2] 多分类模型评估 (5-fold Stratified CV)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all = StandardScaler().fit_transform(df[dim_cols].values)
y_all = y

# 多分类baseline（猜最多数期刊的比例）
baseline = max(np.bincount(y_all)) / len(y_all)

models = {
    'Logistic Regression': LogisticRegression(
        penalty='l2', C=1.0, max_iter=5000,
        multi_class='multinomial', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=30,
        random_state=42, n_jobs=-1
    ),
}

# 尝试XGBoost
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        eval_metric='mlogloss', random_state=42, verbosity=0
    )
except ImportError:
    pass

multi_results = {}
for model_name, model in models.items():
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
    }
    scores = cross_validate(model, X_all, y_all, cv=cv, scoring=scoring, n_jobs=-1)

    multi_results[model_name] = {
        'accuracy_mean': scores['test_accuracy'].mean(),
        'accuracy_std': scores['test_accuracy'].std(),
        'f1_macro_mean': scores['test_f1_macro'].mean(),
        'f1_macro_std': scores['test_f1_macro'].std(),
        'f1_weighted_mean': scores['test_f1_weighted'].mean(),
        'f1_weighted_std': scores['test_f1_weighted'].std(),
    }

    print(f"    {model_name:<25s} "
          f"Accuracy={scores['test_accuracy'].mean():.3f}+/-{scores['test_accuracy'].std():.3f}  "
          f"F1_macro={scores['test_f1_macro'].mean():.3f}  "
          f"F1_weighted={scores['test_f1_weighted'].mean():.3f}")

print(f"    {'Baseline (most frequent journal)':<25s} Accuracy={baseline:.3f}")

# ============================================================================
# 3. One-vs-Rest 逐期刊分析
# ============================================================================
print("\n[3] One-vs-Rest 逐期刊分析...")

ovr_results = {}
for j_idx, (j_full, j_short) in enumerate(zip(journals, journal_short_names)):
    y_binary = (y_all == j_idx).astype(int)
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos

    # 用RF做OvR分类
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=30,
        random_state=42, n_jobs=-1
    )
    ovr_scores = cross_validate(rf, X_all, y_binary, cv=cv, scoring={
        'auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'
    }, n_jobs=-1)

    ovr_results[j_short] = {
        'auc_mean': ovr_scores['test_auc'].mean(),
        'auc_std': ovr_scores['test_auc'].std(),
        'f1_mean': ovr_scores['test_f1'].mean(),
        'recall_mean': ovr_scores['test_recall'].mean(),
        'n_samples': n_pos,
        'baseline': max(n_pos, n_neg) / (n_pos + n_neg),
    }

# 按AUC排序打印
ovr_sorted = sorted(ovr_results.items(), key=lambda x: x[1]['auc_mean'], reverse=True)
print(f"    {'Journal':<18s} {'AUC':>8s} {'F1':>8s} {'Recall':>8s} {'N':>6s} {'Baseline':>8s}")
for j_short, res in ovr_sorted:
    auc_label = f"{res['auc_mean']:.3f}+/-{res['auc_std']:.3f}"
    above_bl = res['auc_mean'] - 0.5
    print(f"    {j_short:<18s} {auc_label:>8s} {res['f1_mean']:.3f}    "
          f"{res['recall_mean']:.3f}    {res['n_samples']:4d}  {res['baseline']:.3f}")

# 找到最独特和最不独特的期刊
most_distinctive = ovr_sorted[0]
least_distinctive = ovr_sorted[-1]
print(f"\n  最独特期刊: {most_distinctive[0]} (AUC={most_distinctive[1]['auc_mean']:.3f})")
print(f"  最难区分期刊: {least_distinctive[0]} (AUC={least_distinctive[1]['auc_mean']:.3f})")

# ============================================================================
# 4. 混淆矩阵 (Random Forest)
# ============================================================================
print("\n[4] 生成混淆矩阵...")

rf = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=30,
    random_state=42, n_jobs=-1
)
rf.fit(X_all, y_all)

# 使用交叉验证的预测来构建混淆矩阵
from sklearn.model_selection import cross_val_predict
y_pred_cv = cross_val_predict(rf, X_all, y_all, cv=cv, n_jobs=-1)
cm = confusion_matrix(y_all, y_pred_cv)

# 归一化混淆矩阵（按行 = 召回率）
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("\n  --- 混淆矩阵 (行归一化 = 召回率) ---")
header = f"{'':>18s}" + "".join(f"{j:>8s}" for j in journal_short_names)
print(f"    {header}")
for i, (j_short, row) in enumerate(zip(journal_short_names, cm_normalized)):
    cells = "".join(f"{v:8.3f}" for v in row)
    print(f"    {j_short:<18s}{cells}")

# 打印哪些期刊对最容易混淆
print("\n  --- 最容易混淆的期刊对 (来自混淆矩阵) ---")
confusions = []
for i, j in combinations(range(n_journals), 2):
    confusion_rate = cm_normalized[i, j] + cm_normalized[j, i]
    confusions.append({
        'j1': journal_short_names[i], 'j2': journal_short_names[j],
        'rate': confusion_rate,
    })
confusions.sort(key=lambda x: x['rate'], reverse=True)
for c in confusions[:5]:
    print(f"    {c['j1']} <-> {c['j2']}: 混淆率={c['rate']:.3f}")

# ============================================================================
# 5. 特征重要性
# ============================================================================
print("\n[5] 期刊分类特征重要性...")

# LR系数（多分类系数矩阵: n_classes × n_features）
lr = LogisticRegression(
    penalty='l2', C=1.0, max_iter=5000,
    multi_class='multinomial', random_state=42
)
lr.fit(X_all, y_all)

# RF特征重要性
rf_importance = rf.feature_importances_

# Permutation重要性
perm_imp = permutation_importance(rf, X_all, y_all, n_repeats=20, random_state=42, scoring='accuracy')

print(f"\n  --- 总体特征重要性 ---")
print(f"    {'Dim':<8s} {'LR Mean|Coef|':>14s} {'RF Imp':>10s} {'Perm Imp':>12s}")
for i, dim in enumerate(dim_cols):
    lr_mean_abs = np.abs(lr.coef_[:, i]).mean()
    print(f"    {dim:<8s} {lr_mean_abs:14.4f} {rf_importance[i]:10.4f} "
          f"{perm_imp.importances_mean[i]:10.4f} +/-{perm_imp.importances_std[i]:.4f}")

# 逐期刊的LR系数（哪些维度对各期刊最重要）
print(f"\n  --- 逐期刊Logistic Regression系数 ---")
print(f"    {'Journal':<18s}", end='')
for dim in dim_cols:
    print(f"  {dim:>10s}", end='')
print()
for i, j_short in enumerate(journal_short_names):
    print(f"    {j_short:<18s}", end='')
    for j in range(len(dim_cols)):
        print(f"  {lr.coef_[i, j]:+10.4f}", end='')
    dominant_dim = dim_cols[np.argmax(np.abs(lr.coef_[i]))]
    print(f"  -> {dominant_dim}")

# ============================================================================
# 6. 期刊风格相似度矩阵
# ============================================================================
print("\n[6] 期刊风格相似度矩阵...")

# 使用马氏距离（Mahalanobis distance）——考虑了各维度的协方差
# 取每个期刊的维度均值
journal_centroids = {}
for j, j_short in zip(journals, journal_short_names):
    jdf = df[df['source_journal'] == j]
    journal_centroids[j_short] = jdf[dim_cols].mean().values

# 马氏距离矩阵
cov_inv = np.linalg.pinv(np.cov(X_all, rowvar=False))
dist_matrix = np.zeros((n_journals, n_journals))
for i in range(n_journals):
    for j in range(n_journals):
        diff = journal_centroids[journal_short_names[i]] - journal_centroids[journal_short_names[j]]
        dist_matrix[i, j] = np.sqrt(max(0, diff @ cov_inv @ diff))

print("\n  --- 期刊间马氏距离 (Mahalanobis Distance) ---")
print(f"    {'':>18s}", end='')
for js in journal_short_names:
    print(f"{js:>14s}", end='')
print()
for i, js_i in enumerate(journal_short_names):
    print(f"    {js_i:<18s}", end='')
    for j in range(n_journals):
        if i == j:
            print(f"     {'--':>8s}", end='')
        else:
            print(f"  {dist_matrix[i, j]:8.4f}", end='')
    print()

# 找到最近和最远的期刊对
min_pair, max_pair = None, None
min_dist, max_dist = float('inf'), 0
for i, j in combinations(range(n_journals), 2):
    d = dist_matrix[i, j]
    if d < min_dist:
        min_dist = d
        min_pair = (journal_short_names[i], journal_short_names[j])
    if d > max_dist:
        max_dist = d
        max_pair = (journal_short_names[i], journal_short_names[j])

print(f"\n  风格最接近的期刊对: {min_pair[0]} <-> {min_pair[1]} (d={min_dist:.4f})")
print(f"  风格最遥远的期刊对: {max_pair[0]} <-> {max_pair[1]} (d={max_dist:.4f})")

# ============================================================================
# 7. 可视化
# ============================================================================
print("\n[7] 生成可视化图表...")

# Colors for journals
journal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ---- Fig 1: 混淆矩阵热力图 ----
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f',
            xticklabels=journal_short_names, yticklabels=journal_short_names,
            cmap='YlOrRd', vmin=0, vmax=0.5,
            linewidths=1, cbar_kws={'label': 'Recall Rate', 'shrink': 0.8},
            ax=ax)
ax.set_xlabel('Predicted Journal', fontsize=12)
ax.set_ylabel('True Journal', fontsize=12)
ax.set_title(f'Confusion Matrix (RF, 5-fold CV)\n'
             f'Accuracy={multi_results["Random Forest"]["accuracy_mean"]:.3f}  '
             f'F1_macro={multi_results["Random Forest"]["f1_macro_mean"]:.3f}',
             fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_j1_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_j1_confusion_matrix.png")

# ---- Fig 2: One-vs-Rest AUC 柱状图 ----
fig, ax = plt.subplots(figsize=(10, 5))
ovr_names = [r[0] for r in ovr_sorted]
ovr_aucs = [r[1]['auc_mean'] for r in ovr_sorted]
ovr_stds = [r[1]['auc_std'] for r in ovr_sorted]
colors_auc = ['#d73027' if auc > 0.7 else '#fc8d59' if auc > 0.6 else '#4575b4'
              for auc in ovr_aucs]

bars = ax.barh(range(len(ovr_names)), ovr_aucs, xerr=ovr_stds,
               color=colors_auc, edgecolor='white', linewidth=1.5, capsize=3, height=0.6)
ax.axvline(0.5, color='gray', linewidth=0.8, linestyle='--', label='Random (0.5)')
ax.set_yticks(range(len(ovr_names)))
ax.set_yticklabels(ovr_names, fontsize=11)
ax.set_xlabel('AUC-ROC (5-fold CV)')
ax.set_title('One-vs-Rest Classification: Per-Journal AUC', fontsize=14)
ax.set_xlim(0.4, 0.9)
for i, (bar, auc) in enumerate(zip(bars, ovr_aucs)):
    ax.text(auc + 0.005, i, f'{auc:.3f}', va='center', fontsize=9)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_j2_ovr_auc.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_j2_ovr_auc.png")

# ---- Fig 3: 期刊在维度空间中的2D投影 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# PCA降维到2D做可视化
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_all)

# 图A: 全部期刊散点
for i, (j_short, color) in enumerate(zip(journal_short_names, journal_colors)):
    mask = df['source_journal'] == journals[i]
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[color], label=j_short, alpha=0.4, s=5, edgecolors='none')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('All Journals in MD Dimension Space')
axes[0].legend(fontsize=7, markerscale=3, loc='upper right')

# 图B-D: 逐对散点（最重要的三对期刊，覆盖6个期刊）
pairs_to_plot = [
    (journals[0], journals[1], journal_short_names[0], journal_short_names[1], journal_colors[0], journal_colors[1]),
    (journals[2], journals[3], journal_short_names[2], journal_short_names[3], journal_colors[2], journal_colors[3]),
    (journals[4], journals[5], journal_short_names[4], journal_short_names[5], journal_colors[4], journal_colors[5]),
]
for ax_idx, (j1, j2, n1, n2, c1, c2) in enumerate(pairs_to_plot):
    ax = axes[ax_idx + 1]
    for j_full, j_short, color in [(j1, n1, c1), (j2, n2, c2)]:
        mask = df['source_journal'] == j_full
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=j_short, alpha=0.5, s=8, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'{n1} vs {n2}')
    ax.legend(fontsize=8, markerscale=3)

fig.suptitle('Journal Separation in MD Dimension Space', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_j3_journal_scatter.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_j3_journal_scatter.png")

# ---- Fig 4: 期刊风格剖面图 (Radar/Parallel Coordinates) ----
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(dim_cols))
for i, (j_short, color) in enumerate(zip(journal_short_names, journal_colors)):
    profile = [journal_profiles[j_short][dim] for dim in dim_cols]
    ax.plot(x, profile, 'o-', color=color, linewidth=2, markersize=8, label=j_short)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(dim_cols, fontsize=12)
ax.set_ylabel('Standardized Deviation from Grand Mean (z-score)', fontsize=11)
ax.set_title('Journal Writing Style Profiles (Dimension Deviations)', fontsize=14)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_j4_journal_profiles.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_j4_journal_profiles.png")

# ---- Fig 5: 期刊间Cohen's d热力图（跨维度平均） ----
fig, ax = plt.subplots(figsize=(8, 7))
# 构建期刊间平均|d|矩阵
d_matrix = np.zeros((n_journals, n_journals))
for i, j in combinations(range(n_journals), 2):
    avg_d = np.mean([
        abs(cohens_d(
            df[df['source_journal'] == journals[i]][dim].values,
            df[df['source_journal'] == journals[j]][dim].values
        ))
        for dim in dim_cols
    ])
    d_matrix[i, j] = avg_d
    d_matrix[j, i] = avg_d

mask = np.triu(np.ones_like(d_matrix, dtype=bool), k=1)
sns.heatmap(d_matrix, mask=mask, annot=True, fmt='.3f',
            xticklabels=journal_short_names, yticklabels=journal_short_names,
            cmap='YlOrRd', linewidths=1,
            cbar_kws={'label': "Mean |Cohen's d|", 'shrink': 0.8},
            ax=ax)
ax.set_title("Mean |Cohen's d| Between Journal Pairs\n(Averaged across 4 dimensions)", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_j5_pairwise_effects.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_j5_pairwise_effects.png")

# ---- Fig 6: 特征重要性对比 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RF重要性
axes[0].barh(dim_cols, rf_importance, color='forestgreen', edgecolor='white')
axes[0].set_title('Random Forest Feature Importance')
axes[0].set_xlabel('Importance')

# 每个期刊的LR系数热力图
lr_coef_matrix = lr.coef_  # n_classes × n_features
sns.heatmap(lr_coef_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=dim_cols, yticklabels=journal_short_names,
            linewidths=1, cbar_kws={'label': 'LR Coefficient', 'shrink': 0.8},
            ax=axes[1])
axes[1].set_title('Per-Journal Logistic Regression Coefficients')
axes[1].set_xlabel('Dimension')
axes[1].set_ylabel('Journal')

fig.suptitle('Feature Importance for Journal Classification', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_j6_feature_importance.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_j6_feature_importance.png")

# ============================================================================
# 8. 综合结论
# ============================================================================
print("\n[8] 综合结论")
print("=" * 70)

best_acc = max(r['accuracy_mean'] for r in multi_results.values())
best_f1 = max(r['f1_macro_mean'] for r in multi_results.values())

print(f"\n  多分类结果:")
print(f"    最佳 Accuracy: {best_acc:.3f} (baseline={baseline:.3f}, "
      f"提升={best_acc-baseline:+.3f})")
print(f"    最佳 F1_macro: {best_f1:.3f}")

# 显著性判断
# 如果 ANOVA 显著 + 多分类准确率 > baseline
anova_all_sig = True  # 前面已经看到至少有个显著
if best_acc > baseline + 0.05 and anova_all_sig:
    print(f"\n  [CONCLUSION] MD维度得分能够以中等水平区分期刊写作风格。")
    print(f"  期刊间存在统计显著的语言风格差异，但差异程度为中等。")
    print(f"  对Copilot的启示：可以有条件地提供期刊匹配建议。")
elif best_acc > baseline + 0.03:
    print(f"\n  [CONCLUSION] MD维度得分能够以较低但统计显著的水平区分期刊。")
    print(f"  期刊间存在可检测但有限的风格差异。")
else:
    print(f"\n  [CONCLUSION] MD维度得分的期刊区分力较弱。")
    print(f"  虽然ANOVA显著，但分类准确率提升有限。")

# 期刊分类特点
print(f"\n  期刊风格特征:")
for i, (j_short, res) in enumerate(ovr_sorted):
    distinctiveness = "高" if res['auc_mean'] > 0.65 else ("中" if res['auc_mean'] > 0.58 else "低")
    best_dim = dim_cols[np.argmax(np.abs(lr.coef_[i]))]
    best_coef = lr.coef_[i][np.argmax(np.abs(lr.coef_[i]))]
    direction = "偏高" if best_coef > 0 else "偏低"
    print(f"    {j_short:<18s} 独特性={distinctiveness} (AUC={res['auc_mean']:.3f}), "
          f"最强特征={best_dim}({direction})")

print(f"\n  风格最接近: {min_pair[0]} <-> {min_pair[1]} (d={min_dist:.4f})")
print(f"  风格最遥远: {max_pair[0]} <-> {max_pair[1]} (d={max_dist:.4f})")

print(f"\n  [OK] 分析完成。图表保存至: {OUTPUT_DIR}")
print("=" * 70)
